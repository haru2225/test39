#!/usr/bin/env python3
"""Periodic full-noise clay CG diffusion experiment based on test38's NequIP.

The forward process is Brownian motion on an orthorhombic periodic cell:
    x_sigma = (x_0 + sigma * Normal(0, I)) mod cell.
Its sigma -> infinity marginal is uniform in the cell. The network predicts
minus sigma times the *wrapped Gaussian conditional score*, not the unwrapped
displacement target used by test38. Sampling starts from independent uniform
positions and uses reverse variance-exploding SDE steps on the same torus.

This is an unconditional model at one composition and fixed cell. It does not
guarantee valid clay structures, equilibrium, chemical bonds, or physical time.
It requires training from scratch; test38 checkpoints use another target.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path

import ase.io
import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list

import test38 as base

FORMAT = "test39-periodic-full-noise-v1"


def coarse_grain(args):
    """Retain selected sites at their original coordinates; omit oxygen/OH sites.

    This is a site-selection CG mapping. It does not combine overlapping SiO4
    and AlO6 groups or silently redistribute their shared oxygen masses.
    """
    positions, cells, source = base.load_dataset(args.dataset)
    names = [species["name"] for species in source["species"]]
    if len(set(names)) != len(names):
        raise ValueError("Dataset species names must be unique")
    removed_names = set(args.remove_species)
    if removed_names - set(names):
        raise ValueError(f"Unknown species to remove: {sorted(removed_names - set(names))}")
    old_types = np.asarray(source["type_ids"], dtype=int)
    kept = np.flatnonzero([names[i] not in removed_names for i in old_types])
    if len(kept) < 2:
        raise ValueError("At least two sites must remain after coarse graining")
    if len(kept) == len(old_types):
        raise ValueError("No sites selected for removal")
    used_types = sorted(set(old_types[kept].tolist()))
    new_ids = {old: new for new, old in enumerate(used_types)}
    meta = copy.deepcopy(source)
    meta["species"] = [copy.deepcopy(source["species"][i]) for i in used_types]
    meta["type_ids"] = [new_ids[int(i)] for i in old_types[kept]]
    description = (
        "Site-selection CG: omit " + ", ".join(sorted(removed_names)) +
        "; keep retained-site coordinates. Export masses are retained atomic-site masses, "
        "not effective group masses. Omitted atoms cannot be reconstructed from this model."
    )
    if "mapping" in meta:
        mapping = meta["mapping"]
        if len(mapping.get("sites", [])) != len(old_types):
            raise ValueError("Source mapping sites do not match the dataset site order")
        mapping["sites"] = [mapping["sites"][int(i)] for i in kept]
        mapping["species"] = copy.deepcopy(meta["species"])
        mapping["description"] = description
    meta["coarse_graining"] = dict(
        method="retained-site-selection", removed_species=sorted(removed_names),
        source_dataset_sha256=source["sha256"],
        source_metadata_sha256=base.digest(args.dataset / "metadata.json"),
        source_sites=len(old_types), retained_sites=len(kept),
        retained_source_site_indices=kept.tolist(), mass_policy="retained-site masses only",
    )
    meta["scientific_caveat"] = str(source.get("scientific_caveat", "")) + " " + description
    if args.reuse and (args.output / "metadata.json").is_file():
        _, _, existing = base.load_dataset(args.output)
        expected = {key: value for key, value in meta.items() if key != "sha256"}
        actual = {key: value for key, value in existing.items() if key != "sha256"}
        if actual != expected:
            raise ValueError("Existing coarse dataset has different source or mapping; choose a new output path")
        print(f"Verified existing coarse dataset: {args.output}", flush=True)
        return 0
    output = base.new_output(args.output)
    reduced = np.lib.format.open_memmap(output / "positions.npy", mode="w+",
                                       dtype=positions.dtype, shape=(len(positions), len(kept), 3))
    for start in range(0, len(positions), 128):
        reduced[start:start + 128] = positions[start:start + 128, kept, :]
    reduced.flush()
    shutil.copyfile(args.dataset / "cells.npy", output / "cells.npy")
    meta["sha256"] = {name: base.digest(output / name) for name in ("positions.npy", "cells.npy")}
    base.save_json(output / "metadata.json", meta)
    ase.io.write(output / "reference.extxyz", base.atoms_from_meta(reduced[0], cells[0], meta))
    print(f"Coarse-grained {len(old_types)} -> {len(kept)} sites, {len(positions)} frames: {output}", flush=True)
    print("Remaining species: " + ", ".join(s["name"] for s in meta["species"]), flush=True)
    return 0


def cell_lengths(cell):
    cell = np.asarray(cell, dtype=float)
    if not np.allclose(cell, np.diag(np.diag(cell)), atol=1e-5) or np.any(np.diag(cell) <= 0):
        raise ValueError("test39 currently requires a positive, axis-aligned orthorhombic cell")
    return np.diag(cell)


def estimate_num_neighbors(positions, cell, type_ids, cutoff):
    """Mean neighbor count within `cutoff` for one frame, used to calibrate the
    NequIP interaction layer's normalization constant to this dataset's actual
    site density (architecture()'s hardcoded default of 12 was tuned for a much
    denser, oxygen-containing CG mapping and can be far off for a sparser one)."""
    data = base.graph(positions, cell, type_ids, cutoff, torch.device("cpu"))
    return float(data.edge_index.shape[1]) / data.num_nodes


def estimate_sigma_data(positions, cell, cutoff=10.0):
    """Median nearest-neighbor distance for one frame, used as the natural
    length scale to center a log-normal training sigma distribution on."""
    i, _, d = primitive_neighbor_list("ijd", [True] * 3, cell, positions, cutoff=cutoff)
    if not len(i):
        raise ValueError("No neighbors within cutoff; cannot estimate a data length scale")
    nearest = np.full(len(positions), np.inf)
    np.minimum.at(nearest, i, d)
    return float(np.median(nearest[np.isfinite(nearest)]))


def wrapped_score_target(noisy, clean, lengths, sigma):
    """Return -sigma * grad log p_sigma(noisy | clean) for Brownian motion on a torus.

    Image sums are stable at small sigma. A Fourier heat-kernel series is stable
    when sigma is large relative to the box; the two overlap near 0.2 L.
    Each Cartesian dimension factorizes for an orthorhombic cell.
    """
    length = torch.as_tensor(lengths, dtype=noisy.dtype, device=noisy.device)
    delta = torch.remainder(noisy - clean + length / 2, length) - length / 2
    result = torch.empty_like(delta)
    for axis in range(3):
        side = length[axis]
        d = delta[:, axis:axis + 1]
        if sigma / float(side) < 0.2:
            # At this switch, omitted |n| >= 2 images have negligible weight.
            n = torch.arange(-1, 2, dtype=noisy.dtype, device=noisy.device)
            images = d + n * side
            weights = torch.softmax(-0.5 * (images / sigma).square(), dim=-1)
            result[:, axis] = (weights * images).sum(dim=-1) / sigma
        else:
            k = torch.arange(1, 13, dtype=noisy.dtype, device=noisy.device)
            amplitude = torch.exp(-2 * math.pi**2 * k.square() * (sigma / side) ** 2)
            angle = 2 * math.pi * d * k / side
            density = 1 + 2 * (amplitude * torch.cos(angle)).sum(dim=-1)
            derivative = -(4 * math.pi / side) * (
                k * amplitude * torch.sin(angle)).sum(dim=-1)
            result[:, axis] = -sigma * derivative / density.clamp_min(1e-8)
    return result


def num_neighbors_value(value):
    if value == "auto":
        return value
    return base.positive(value)


def schedule(sigma_min, sigma_max, steps, device):
    levels = torch.exp(torch.linspace(math.log(sigma_max), math.log(sigma_min), steps,
                                      device=device, dtype=torch.float64))
    return torch.cat((levels, levels.new_zeros(1)))


def write_xyz(output, trajectory, valid, cell, meta):
    path = output / "generation.xyz"
    temporary = output / "generation.xyz.tmp"
    with temporary.open("w") as stream:
        for step in range(valid):
            atoms = base.atoms_from_meta(np.asarray(trajectory[step]), cell, meta)
            atoms.info.update(generation_step=step,
                              initial_distribution="uniform_periodic",
                              is_equilibrium_trajectory=False)
            ase.io.write(stream, atoms, format="extxyz")
    temporary.replace(path)


def settings_for(args, meta, lengths, sigma_max, num_neighbors=None, sigma_log=None, capacity=None):
    settings = dict(dataset_sha256=meta["sha256"], sigma_min=args.sigma_min,
                sigma_max=sigma_max, cutoff=args.cutoff, batch_size=args.batch_size,
                learning_rate=args.learning_rate, seed=args.seed, device=str(args.device),
                cell_lengths_A=np.asarray(lengths).tolist())
    # Only recorded when explicitly requested, so a checkpoint trained before
    # these options existed (default --num-neighbors/--sigma-sampling/
    # --irreps-hidden/--num-convs) resumes against an identical settings dict --
    # adding these keys unconditionally would break --resume for any
    # already-running job.
    if num_neighbors is not None:
        settings["num_neighbors"] = num_neighbors
    if sigma_log is not None:
        settings["sigma_sampling"] = "log-normal"
        settings["sigma_log_mean"], settings["sigma_log_std"] = sigma_log
    if capacity is not None:
        settings["irreps_hidden"], settings["num_convs"] = capacity
    return settings


def train(args):
    positions, cells, meta = base.load_dataset(args.dataset)
    if len(positions) < 3:
        raise ValueError("At least 3 dataset frames are needed")
    lengths = cell_lengths(cells[0])
    for cell in cells:
        if not np.allclose(cell_lengths(cell), lengths, atol=1e-4):
            raise ValueError("test39 currently requires a fixed cell across frames")
    sigma_max = float(args.sigma_max or max(lengths))
    if sigma_max <= args.sigma_min:
        raise ValueError("sigma-max must exceed sigma-min")
    # The first periodic Fourier mode must be nearly extinguished before a
    # uniform prior can approximate the forward terminal distribution.
    residual = math.exp(-2 * math.pi**2 * (sigma_max / max(lengths)) ** 2)
    if residual > 1e-5:
        raise ValueError(f"Terminal noise is not near uniform (Fourier residual={residual:.3g}); "
                         f"use sigma-max >= {math.sqrt(-math.log(1e-5)/(2*math.pi**2))*max(lengths):.2f} A")
    device = base.device_for(args.device)
    output = args.output.resolve() if args.resume else base.new_output(args.output)
    path = output / "checkpoint.pt"
    config = base.architecture(len(meta["species"]), args.cutoff)
    num_neighbors_override = None
    if args.num_neighbors is not None:
        if args.num_neighbors == "auto":
            num_neighbors_override = estimate_num_neighbors(
                positions[0], cells[0], meta["type_ids"], args.cutoff)
        else:
            num_neighbors_override = float(args.num_neighbors)
        config["num_neighbors"] = num_neighbors_override
    capacity_override = None
    if args.irreps_hidden is not None or args.num_convs is not None:
        # architecture()'s irreps_hidden="64x0e + 32x1e" / num_convs=3 were sized
        # for test38's local, small-perturbation denoising task. Generating a
        # whole structure from complete uniform noise is a much harder problem
        # (see TEST39.md); this network is likely undersized for it regardless
        # of the sigma-sampling/num-neighbors fixes above. Widening/deepening it
        # is a first, low-risk capacity increase before considering a different
        # architecture family entirely.
        capacity_override = (args.irreps_hidden or config["irreps_hidden"],
                             args.num_convs or config["num_convs"])
        config["irreps_hidden"], config["num_convs"] = capacity_override
    sigma_log = None
    if args.sigma_sampling == "log-normal":
        log_mean = (math.log(estimate_sigma_data(positions[0], cells[0], args.cutoff))
                    if args.sigma_log_mean is None else math.log(args.sigma_log_mean))
        sigma_log = (log_mean, args.sigma_log_std)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = base.build_time_model(config, device, args.gradient_checkpointing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    settings = settings_for(args, meta, lengths, sigma_max, num_neighbors_override, sigma_log, capacity_override)
    completed, history = 0, []
    if args.resume:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("format") != FORMAT or ck["settings"] != settings:
            raise ValueError("Resume requires an identical test39 checkpoint and settings")
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        base.restore_rng(ck["rng"])
        completed, history = ck["completed_updates"], ck["history"]
    split = max(2, int(len(positions) * 0.9))
    deadline = time.monotonic() + args.time_budget_hours * 3600
    lengths_t = torch.tensor(lengths, dtype=torch.float32, device=device)

    def save():
        base.save_checkpoint(path, dict(
            format=FORMAT, settings=settings, architecture=config,
            model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), rng=base.rng_state(), history=history,
            completed_updates=completed, requested_updates=args.updates,
            dataset_metadata=meta, cell_angstrom=np.asarray(cells[0]).copy(),
            scientific_caveat=__doc__,
        ))
        base.save_json(output / "training.json", dict(
            completed_updates=completed, requested_updates=args.updates,
            terminal_fourier_residual=residual, history=history))

    def loss_for(indices, sigma):
        examples = []
        targets = []
        for i in indices:
            clean = torch.tensor(np.asarray(positions[i]).copy(), dtype=torch.float32, device=device)
            noisy = torch.remainder(clean + sigma * torch.randn_like(clean), lengths_t)
            examples.append(base.graph(noisy.detach().cpu().numpy(), cells[i], meta["type_ids"],
                                       args.cutoff, device))
            targets.append(wrapped_score_target(noisy, clean, lengths_t, sigma))
        batch = base.Batch.from_data_list(examples)
        t = torch.tensor([sigma / sigma_max], dtype=torch.float32, device=device)
        prediction = model(batch, t)
        return torch.nn.functional.mse_loss(prediction, torch.cat(targets, dim=0))

    print(f"test39 train: frames={len(positions)}, sites={len(meta['type_ids'])}, "
          f"sigma={args.sigma_min:g}..{sigma_max:g} A, "
          f"terminal Fourier residual={residual:.2g}, "
          f"num_neighbors={config['num_neighbors']:g}, "
          f"irreps_hidden={config['irreps_hidden']!r}, num_convs={config['num_convs']}, "
          f"sigma_sampling={args.sigma_sampling}" +
          (f" (log-mean={sigma_log[0]:.3g}, log-std={sigma_log[1]:.3g})" if sigma_log else ""),
          flush=True)
    validation_indices = list(range(split, min(split + args.validation_frames, len(positions))))
    for step in range(completed + 1, args.updates + 1):
        if base.STOP or time.monotonic() >= deadline:
            save()
            print("Training paused; resume with --resume", flush=True)
            return 75
        if sigma_log is not None:
            # Log-normal, centered on this dataset's own nearest-neighbor scale:
            # spends most updates where the score actually has structure to learn,
            # instead of spreading them uniformly across three decades of sigma.
            log_sigma = float(np.clip(np.random.normal(*sigma_log),
                                      math.log(args.sigma_min), math.log(sigma_max)))
            sigma = math.exp(log_sigma)
        else:
            # Log-uniform sigma gives even coverage across small and large scales.
            sigma = math.exp(np.random.uniform(math.log(args.sigma_min), math.log(sigma_max)))
        indices = np.random.randint(split, size=args.batch_size)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(indices, sigma)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        completed = step
        if step == 1 or step % args.log_every == 0 or step == args.updates:
            model.eval()
            saved_rng = base.rng_state()
            with torch.no_grad():
                diagnostics = {}
                for label, s in (("small", args.sigma_min * 4),
                                 ("middle", math.sqrt(args.sigma_min * sigma_max)),
                                 ("terminal", sigma_max)):
                    # Averaged over several held-out frames so a single-frame
                    # fluke doesn't read as a real change in model quality.
                    diagnostics[label] = float(loss_for(validation_indices, min(s, sigma_max)).item())
            base.restore_rng(saved_rng)
            row = dict(step=step, train_mse=float(loss.detach().cpu()),
                       sigma_A=sigma, validation_mse=diagnostics)
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.checkpoint_every == 0:
            save()
    save()
    print(f"Checkpoint: {path}", flush=True)
    return 0


@torch.no_grad()
def generate(args):
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("format") != FORMAT:
        raise ValueError("A trained test39 checkpoint is required; test38 checkpoints are incompatible")
    if args.deterministic_steps > args.reverse_steps:
        raise ValueError("deterministic-steps must not exceed reverse-steps")
    device = base.device_for(args.device)
    model = base.build_time_model(ck["architecture"], device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    lengths = torch.tensor(ck["settings"]["cell_lengths_A"], dtype=torch.float32, device=device)
    cell = ck["cell_angstrom"]
    meta = ck["dataset_metadata"]
    sigma_min, sigma_max = ck["settings"]["sigma_min"], ck["settings"]["sigma_max"]
    output = args.output.resolve() if args.resume else base.new_output(args.output)
    levels = schedule(sigma_min, sigma_max, args.reverse_steps, device)
    settings = dict(checkpoint_sha256=base.digest(args.checkpoint), reverse_steps=args.reverse_steps,
                    deterministic_steps=args.deterministic_steps, seed=args.seed, device=str(device),
                    initial_state="uniform_periodic")
    state_path = output / "generation_restart.pt"
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["settings"] != settings:
            raise ValueError("Generation restart settings or checkpoint changed")
        pos, completed = state["positions"].to(device), state["step"]
        base.restore_rng(state["rng"])
        trajectory = np.lib.format.open_memmap(output / "positions.npy", mode="r+")
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        pos = torch.rand((len(meta["type_ids"]), 3), dtype=torch.float32, device=device) * lengths
        completed = 0
        trajectory = np.lib.format.open_memmap(output / "positions.npy", mode="w+",
                                               dtype=np.float32, shape=(args.reverse_steps + 1, len(pos), 3))
        trajectory[0] = pos.cpu().numpy()
    deadline = time.monotonic() + args.time_budget_hours * 3600

    def save(write_movie=False):
        trajectory.flush()
        base.save_checkpoint(state_path, dict(settings=settings, step=completed,
                                              positions=pos.detach().cpu(), rng=base.rng_state()))
        base.save_json(output / "generation.json", dict(
            format=FORMAT, complete=completed == args.reverse_steps,
            valid_frames=completed + 1, completed_steps=completed,
            requested_steps=args.reverse_steps, settings=settings,
            sigma_min=sigma_min, sigma_max=sigma_max,
            cell_angstrom=np.asarray(cell).tolist(), dataset_metadata=meta,
            initial_distribution="uniform positions in periodic cell",
            is_equilibrium_trajectory=False))
        if write_movie:
            write_xyz(output, trajectory, completed + 1, cell, meta)

    for step in range(completed, args.reverse_steps):
        if base.STOP or time.monotonic() >= deadline:
            save(write_movie=True)
            print("Generation paused; resume with --resume", flush=True)
            return 75
        sigma = float(levels[step])
        variance_drop = float(levels[step] ** 2 - levels[step + 1] ** 2)
        data = base.graph(pos.cpu().numpy(), cell, meta["type_ids"],
                          ck["architecture"]["cutoff_angstrom"], device)
        scaled_negative_score = model(data, torch.tensor([sigma / sigma_max], device=device))
        # Reverse SDE: drift = delta_variance * score; score = -model / sigma.
        multiplier = 0.5 if step >= args.reverse_steps - args.deterministic_steps else 1.0
        pos = pos - multiplier * (variance_drop / sigma) * scaled_negative_score
        if multiplier == 1.0:
            pos = pos + math.sqrt(variance_drop) * torch.randn_like(pos)
        pos = torch.remainder(pos, lengths)
        if not torch.isfinite(pos).all():
            raise RuntimeError("Non-finite generated positions")
        completed = step + 1
        trajectory[completed] = pos.cpu().numpy()
        if completed % args.checkpoint_every == 0:
            save()
            print(f"generation {completed}/{args.reverse_steps}, sigma={sigma:.4g} A", flush=True)
    save(write_movie=True)
    atoms = base.atoms_from_meta(pos.cpu().numpy(), cell, meta)
    ase.io.write(output / "final.extxyz", atoms)
    print(f"Generated {completed + 1} frames: {output}", flush=True)
    return 0


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="stage", required=True)
    cg_p = sub.add_parser("coarse-grain", help="Remove clay oxygen/OH sites from a prepared dataset")
    cg_p.add_argument("--dataset", type=Path, required=True)
    cg_p.add_argument("--output", type=Path, required=True)
    cg_p.add_argument("--reuse", action="store_true", help="Reuse only a verified dataset with the same source and selection")
    cg_p.add_argument("--remove-species", nargs="+", default=["ob", "obos", "oh", "ohs", "ho"],
                      help="Species labels to omit; default removes clay O and hydroxyl H")
    cg_p.set_defaults(handler=coarse_grain)
    train_p = sub.add_parser("train")
    train_p.add_argument("--dataset", type=Path, required=True)
    train_p.add_argument("--updates", type=base.count, default=30000)
    train_p.add_argument("--batch-size", type=base.count, default=2)
    train_p.add_argument("--learning-rate", type=base.positive, default=2e-4)
    train_p.add_argument("--cutoff", type=base.positive, default=10.0)
    train_p.add_argument("--sigma-min", type=base.positive, default=0.05)
    train_p.add_argument("--sigma-max", type=base.positive, default=None,
                         help="Default: longest cell edge; must yield a near-uniform terminal state")
    train_p.add_argument("--log-every", type=base.count, default=100)
    train_p.add_argument("--num-neighbors", type=num_neighbors_value, default=None,
                         help="Override architecture()'s hardcoded 12 (tuned for a denser, "
                              "oxygen-containing CG mapping); pass 'auto' to estimate the mean "
                              "neighbor count within --cutoff from the dataset's first frame, "
                              "or a numeric value directly. Default: keep the old 12 (unchanged, "
                              "so this does not affect resuming an existing checkpoint)")
    train_p.add_argument("--sigma-sampling", choices=("log-uniform", "log-normal"), default="log-uniform",
                         help="log-uniform (default, unchanged) spreads training sigma evenly "
                              "across sigma-min..sigma-max in log space. log-normal instead "
                              "concentrates updates near --sigma-log-mean (default: this "
                              "dataset's own median nearest-neighbor distance), which is where "
                              "the score actually has learnable structure -- sigma-max itself "
                              "still needs to stay large enough for a uniform terminal state; "
                              "only how densely intermediate sigmas are sampled changes")
    train_p.add_argument("--sigma-log-mean", type=base.positive, default=None,
                         help="log-normal sigma sampling only: center of the sigma distribution "
                              "in angstrom. Default: estimated median nearest-neighbor distance")
    train_p.add_argument("--sigma-log-std", type=base.positive, default=1.2,
                         help="log-normal sigma sampling only: spread in natural-log space "
                              "(EDM-style default of 1.2 reaches roughly a factor of 25 either "
                              "side of --sigma-log-mean before clipping to [sigma-min, sigma-max])")
    train_p.add_argument("--validation-frames", type=base.count, default=8,
                         help="Held-out frames averaged into each logged validation_mse "
                              "(small/middle/terminal), to smooth out single-frame noise")
    train_p.add_argument("--irreps-hidden", type=str, default=None,
                         help="Override architecture()'s hardcoded '64x0e + 32x1e' hidden "
                              "irreps, e.g. '128x0e + 64x1e + 32x2e'. That default was sized "
                              "for test38's small local denoising task; generating a whole "
                              "structure from complete uniform noise likely needs more "
                              "capacity. Default: keep the old value (unchanged, so this does "
                              "not affect resuming an existing checkpoint)")
    train_p.add_argument("--num-convs", type=base.count, default=None,
                         help="Override architecture()'s hardcoded 3 interaction layers, e.g. "
                              "5 or 6. More layers let more angular/many-body correlation "
                              "implicitly build up through successive tensor-product mixing "
                              "(NequIP has no explicit 3-body term). Default: keep the old "
                              "value 3 (unchanged, does not affect --resume)")
    train_p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                         help="Recompute each Interaction+Gate layer's forward during backward "
                              "instead of keeping it in memory, to reduce peak GPU memory (no "
                              "effect on the model's output, capacity or saved weights -- only a "
                              "compute/memory tradeoff). Default on, since the widened/l=5 "
                              "architecture can otherwise hit CUDA out-of-memory; pass "
                              "--no-gradient-checkpointing to disable")
    train_p.set_defaults(handler=train)
    gen_p = sub.add_parser("generate")
    gen_p.add_argument("--checkpoint", type=Path, required=True)
    gen_p.add_argument("--reverse-steps", type=base.count, default=300)
    gen_p.add_argument("--deterministic-steps", type=base.nonnegative_count, default=1)
    gen_p.set_defaults(handler=generate)
    for command in (train_p, gen_p):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="auto")
        command.add_argument("--seed", type=int, default=1337)
        command.add_argument("--resume", action="store_true")
        command.add_argument("--time-budget-hours", type=base.positive, default=11.5)
        command.add_argument("--checkpoint-every", type=base.count, default=25)
    return root


def main():
    import signal
    signal.signal(signal.SIGTERM, base.request_stop)
    signal.signal(signal.SIGINT, base.request_stop)
    args = parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
