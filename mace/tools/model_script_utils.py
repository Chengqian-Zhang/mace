import ast
import logging

import numpy as np
import torch
from e3nn import o3

from mace import modules
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools.finetuning_utils import load_foundations_elements
from mace.tools.scripts_utils import extract_config_mace_model
from mace.tools.torch_tools import dtype_dict
from mace.tools.utils import AtomicNumberTable


def configure_model(
    args,
    train_loader,
    atomic_energies,
    model_foundation=None,
    heads=None,
    z_table=None,
    head_configs=None,
    property_head_configs=None,
    property_mean=None,
    property_std=None,
):
    # Selecting outputs
    compute_virials = args.loss == "virials"
    compute_stress = args.loss in ("stress", "huber", "universal")

    if compute_virials:
        args.compute_virials = True
    elif compute_stress:
        args.compute_stress = True

    if compute_virials or compute_stress:
        if args.error_table in ["PerAtomRMSE", "PerAtomMAE", "TotalRMSE", "TotalMAE"]:
            args.error_table = (
                "PerAtomRMSEstressvirials"
                if "RMSE" in args.error_table
                else "PerAtomMAEstressvirials"
            )

    output_args = {
        "energy": args.compute_energy,
        "forces": args.compute_forces,
        "virials": compute_virials,
        "stress": compute_stress,
        "dipoles": args.compute_dipole,
        "polarizabilities": args.compute_polarizability,
    }
    logging.info(
        f"During training the following quantities will be reported: {', '.join([f'{report}' for report, value in output_args.items() if value])}"
    )
    logging.info("===========MODEL DETAILS===========")

    if args.scaling == "no_scaling":
        args.std = 1.0
        if head_configs is not None:
            for head_config in head_configs:
                head_config.std = 1.0
        logging.info("No scaling selected")

    if (
        head_configs is not None
        and args.std is not None
        and not isinstance(args.std, list)
    ):
        atomic_inter_scale = []
        for head_config in head_configs:
            if hasattr(head_config, "std") and head_config.std is not None:
                atomic_inter_scale.append(head_config.std)
            elif args.std is not None:
                atomic_inter_scale.append(
                    args.std if isinstance(args.std, float) else 1.0
                )
        args.std = atomic_inter_scale

    elif (args.mean is None or args.std is None) and (
        args.model not in ("AtomicDipolesMACE", "AtomicDielectricMACE")
    ):
        args.mean, args.std = modules.scaling_classes[args.scaling](
            train_loader, atomic_energies
        )
    if args.embedding_specs is not None:
        logging.info("Using embedding specifications from command line arguments")
        logging.info(f"Embedding specifications: {args.embedding_specs}")
    # Build model
    if model_foundation is not None and args.model in [
        "MACE",
        "ScaleShiftMACE",
        "ScaleShiftMACEProperty",
        "MACELES",
        "PolarMACE",
    ]:
        logging.info("Loading FOUNDATION model")
        model_config_foundation = extract_config_mace_model(model_foundation)
        model_config_foundation["atomic_energies"] = atomic_energies

        if args.foundation_model_elements:
            foundation_z_table = AtomicNumberTable(
                [int(z) for z in model_foundation.atomic_numbers]
            )
            model_config_foundation["atomic_numbers"] = foundation_z_table.zs
            model_config_foundation["num_elements"] = len(foundation_z_table)
            z_table = foundation_z_table
            logging.info(
                f"Using all elements from foundation model: {foundation_z_table.zs}"
            )
        else:
            model_config_foundation["atomic_numbers"] = z_table.zs
            model_config_foundation["num_elements"] = len(z_table)
            logging.info(f"Using filtered elements: {z_table.zs}")

        args.max_L = model_config_foundation["hidden_irreps"].lmax

        if args.model in (
            "ScaleShiftMACE",
            "PolarMACE",
        ) or model_foundation.__class__.__name__ in (
            "ScaleShiftMACE",
            "PolarMACE",
        ):
            model_config_foundation["atomic_inter_shift"] = (
                _determine_atomic_inter_shift(args.mean, heads)
            )
        else:
            model_config_foundation["atomic_inter_shift"] = [0.0] * len(heads)
        model_config_foundation["atomic_inter_scale"] = [1.0] * len(heads)
        if getattr(args, "avg_num_neighbors", None) is not None:
            model_config_foundation["avg_num_neighbors"] = args.avg_num_neighbors
        else:
            args.avg_num_neighbors = model_config_foundation["avg_num_neighbors"]
        if args.model == "MACELES":
            args.model = "FoundationMACELES"
        elif args.model in ("MACE", "ScaleShiftMACE"):
            args.model = "FoundationMACE"
        model_config_foundation["heads"] = heads
        model_config = model_config_foundation

        logging.info("Model configuration extracted from foundation model")
        logging.info(f"Using {args.loss} loss function for fine-tuning")
        logging.info(
            f"Message passing with hidden irreps {model_config_foundation['hidden_irreps']})"
        )
        logging.info(
            f"{model_config_foundation['num_interactions']} layers, each with correlation order: {model_config_foundation['correlation']} (body order: {model_config_foundation['correlation']+1}) and spherical harmonics up to: l={model_config_foundation['max_ell']}"
        )
        logging.info(
            f"Radial cutoff: {model_config_foundation['r_max']} A (total receptive field for each atom: {model_config_foundation['r_max'] * model_config_foundation['num_interactions']} A)"
        )
        logging.info(
            f"Distance transform for radial basis functions: {model_config_foundation['distance_transform']}"
        )
    else:
        logging.info("Building model")
        logging.info(
            f"Message passing with {args.num_channels} channels and max_L={args.max_L} ({args.hidden_irreps})"
        )
        logging.info(
            f"{args.num_interactions} layers, each with correlation order: {args.correlation} (body order: {args.correlation+1}) and spherical harmonics up to: l={args.max_ell}"
        )
        logging.info(
            f"{args.num_radial_basis} radial and {args.num_cutoff_basis} basis functions"
        )
        logging.info(
            f"Radial cutoff: {args.r_max} A (total receptive field for each atom: {args.r_max * args.num_interactions} A)"
        )
        logging.info(
            f"Distance transform for radial basis functions: {args.distance_transform}"
        )

        assert (
            len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
        ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"

        logging.info(f"Hidden irreps: {args.hidden_irreps}")

        cueq_config = None
        if args.only_cueq:
            logging.info("Using only the backend of the model")
            cueq_config = CuEquivarianceConfig(
                enabled=True,
                layout="ir_mul",
                group="O3_e3nn",
                optimize_all=True,
                conv_fusion=(args.device == "cuda"),
            )

        model_config = dict(
            r_max=args.r_max,
            num_bessel=args.num_radial_basis,
            num_polynomial_cutoff=args.num_cutoff_basis,
            max_ell=args.max_ell,
            interaction_cls=modules.interaction_classes[args.interaction],
            num_interactions=args.num_interactions,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps(args.hidden_irreps),
            edge_irreps=o3.Irreps(args.edge_irreps) if args.edge_irreps else None,
            atomic_energies=atomic_energies,
            apply_cutoff=args.apply_cutoff,
            avg_num_neighbors=args.avg_num_neighbors,
            atomic_numbers=z_table.zs,
            use_reduced_cg=args.use_reduced_cg,
            use_so3=args.use_so3,
            use_edge_irreps_first=args.use_edge_irreps_first,
            cueq_config=cueq_config,
        )
        model_config_foundation = None

    model = _build_model(
        args,
        model_config,
        model_config_foundation,
        heads,
        property_head_configs=property_head_configs,
        property_mean=property_mean,
        property_std=property_std,
        property_head_seed=getattr(args, "seed", None),
    )

    if model_foundation is not None:
        target = model.backbone if hasattr(model, "backbone") else model
        # In MFT, z_table is the union of all head species (property + energy), so
        # len(z_table) > len(property_head z_table). Use only property-head species
        # for the node-embedding scaling divisor to match single-task finetune scale.
        reference_num_species = None
        if property_head_configs:
            prop_zs: set = set()
            for hc in property_head_configs:
                prop_zs.update(hc.atomic_numbers)
            reference_num_species = len(prop_zs)
        loaded = load_foundations_elements(
            target,
            model_foundation,
            z_table,
            load_readout=args.foundation_filter_elements,
            max_L=args.max_L,
            default_dtype=dtype_dict.get(args.default_dtype, torch.float64),
            reference_num_species=reference_num_species,
            avg_num_neighbors=args.avg_num_neighbors,
        )
        if hasattr(model, "backbone"):
            model.backbone = loaded
        else:
            model = loaded

    return model, output_args


def _determine_atomic_inter_shift(mean, heads):
    if isinstance(mean, np.ndarray):
        if mean.size == 1:
            return mean.item()
        if mean.size == len(heads):
            return mean.tolist()
        logging.info("Mean not in correct format, using default value of 0.0")
        return [0.0] * len(heads)
    if isinstance(mean, list) and len(mean) == len(heads):
        return mean
    if isinstance(mean, float):
        return [mean] * len(heads)
    logging.info("Mean not in correct format, using default value of 0.0")
    return [0.0] * len(heads)


def _parse_literal_or_none(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in ("none", ""):
            return None
        return ast.literal_eval(stripped)
    return value


def _build_model(
    args, model_config, model_config_foundation, heads, property_head_configs=None,
    property_mean=None, property_std=None, property_head_seed=None,
):  # pylint: disable=too-many-return-statements
    if args.model == "MACE":
        if args.interaction_first not in [
            "RealAgnosticInteractionBlock",
            "RealAgnosticDensityInteractionBlock",
        ]:
            args.interaction_first = "RealAgnosticInteractionBlock"
        return modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=[0.0] * len(heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            embedding_specs=args.embedding_specs,
            use_embedding_readout=args.use_embedding_readout,
            use_last_readout_only=args.use_last_readout_only,
            use_agnostic_product=args.use_agnostic_product,
        )
    if args.model == "ScaleShiftMACE":
        return modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=args.mean,
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            embedding_specs=args.embedding_specs,
            use_embedding_readout=args.use_embedding_readout,
            use_last_readout_only=args.use_last_readout_only,
            use_agnostic_product=args.use_agnostic_product,
        )
    if args.model == "PolarMACE" and model_config_foundation is not None:
        return modules.PolarMACE(**model_config_foundation)
    if args.model == "PolarMACE":
        field_feature_widths = _parse_literal_or_none(args.field_feature_widths)
        field_feature_norms = _parse_literal_or_none(args.field_feature_norms)
        fixedpoint_update_config = _parse_literal_or_none(args.fixedpoint_update_config)
        field_readout_config = _parse_literal_or_none(args.field_readout_config)
        return modules.PolarMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=_determine_atomic_inter_shift(args.mean, heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            embedding_specs=args.embedding_specs,
            use_embedding_readout=args.use_embedding_readout,
            use_last_readout_only=args.use_last_readout_only,
            use_agnostic_product=args.use_agnostic_product,
            kspace_cutoff_factor=args.kspace_cutoff_factor,
            atomic_multipoles_max_l=args.atomic_multipoles_max_l,
            atomic_multipoles_smearing_width=args.atomic_multipoles_smearing_width,
            field_feature_max_l=args.field_feature_max_l,
            field_feature_widths=(
                field_feature_widths if field_feature_widths is not None else [1.0]
            ),
            num_recursion_steps=args.num_recursion_steps,
            field_si=args.field_si,
            include_electrostatic_self_interaction=args.include_electrostatic_self_interaction,
            add_local_electron_energy=args.add_local_electron_energy,
            quadrupole_feature_corrections=args.quadrupole_feature_corrections,
            return_electrostatic_potentials=args.return_electrostatic_potentials,
            field_feature_norms=field_feature_norms,
            field_norm_factor=args.field_norm_factor,
            fixedpoint_update_config=fixedpoint_update_config,
            field_readout_config=field_readout_config,
        )
    if args.model == "ScaleShiftMACEProperty":
        # Build backbone as ScaleShiftMACE, then wrap with property readout heads.
        # model_config_foundation is set when --foundation_model is provided;
        # otherwise fall back to model_config (built from --hidden_irreps etc).
        if model_config_foundation is not None:
            backbone = modules.ScaleShiftMACE(**model_config_foundation)
        else:
            backbone = modules.ScaleShiftMACE(
                **model_config,
                pair_repulsion=args.pair_repulsion,
                distance_transform=args.distance_transform,
                correlation=args.correlation,
                gate=modules.gate_dict[args.gate],
                interaction_cls_first=modules.interaction_classes[args.interaction_first],
                MLP_irreps=o3.Irreps(args.MLP_irreps),
                atomic_inter_scale=args.std,
                atomic_inter_shift=args.mean,
                radial_MLP=ast.literal_eval(args.radial_MLP),
                radial_type=args.radial_type,
                heads=heads,
                embedding_specs=args.embedding_specs,
                use_embedding_readout=args.use_embedding_readout,
                use_last_readout_only=args.use_last_readout_only,
                use_agnostic_product=args.use_agnostic_product,
            )

        # hidden_irreps determines the per-layer feature dim and scalar count
        raw_irreps = (
            model_config_foundation.get("hidden_irreps", args.hidden_irreps)
            if model_config_foundation is not None
            else model_config.get("hidden_irreps", args.hidden_irreps)
        )
        hidden_irreps = o3.Irreps(str(raw_irreps))
        node_feats_scalar_dim = hidden_irreps.count(o3.Irrep(0, 1))
        node_feats_hidden_dim = hidden_irreps.dim
        # l=1 equivariant features per layer (used for mu aggregation)
        n_l1 = sum(mul for mul, ir in hidden_irreps if ir.l == 1)
        node_feats_vector_dim = n_l1 * 3  # 3 spatial components per l=1 channel

        # Collect property head info from head_configs or args
        if property_head_configs:
            prop_head_names = [hc.head_name for hc in property_head_configs]
            task_dims = {hc.head_name: hc.task_dim for hc in property_head_configs}
            prop_intensive = {
                hc.head_name: (True if hc.property_intensive is None else hc.property_intensive)
                for hc in property_head_configs
            }
            prop_aggregation = {
                hc.head_name: (hc.property_aggregation or "default")
                for hc in property_head_configs
            }
        else:
            prop_name = getattr(args, "property_name", None) or "property_head"
            prop_head_names = [prop_name]
            task_dims = {prop_name: getattr(args, "task_dim", 1) or 1}
            prop_intensive = {prop_name: getattr(args, "property_intensive", True)}
            prop_aggregation = {prop_name: getattr(args, "property_aggregation", "default")}

        # For mu aggregation: task_dim = n_l1 * n_full_layers (auto-set, not user-specified).
        # n_full_layers = layers with full hidden_irreps (have l=1 features).
        # ScaleShiftMACE: last layer uses hidden_irreps[0] (l=0 only) unless keep_last_layer_irreps.
        # For n_interactions==1 the single layer also gets l=0-only treatment → n_full_layers=0.
        # Assumes hidden_irreps lists 0e before 1o (standard MACE), so l=1 offset = scalar_dim.
        if n_l1 > 0:
            n_interactions = int(backbone.num_interactions)
            keep_last = False
            if model_config_foundation is not None:
                keep_last = model_config_foundation.get("keep_last_layer_irreps", False)
            else:
                keep_last = model_config.get("keep_last_layer_irreps", False)
            n_full_layers = n_interactions if keep_last else n_interactions - 1
            for head_name in prop_head_names:
                if prop_aggregation.get(head_name, "default") == "mu":
                    if n_full_layers <= 0:
                        raise ValueError(
                            f"property_aggregation='mu' requires num_interactions >= 2 "
                            f"(got {n_interactions}). Use at least 2 interaction layers."
                        )
                    task_dims[head_name] = n_l1 * n_full_layers

        # Decouple property head init from backbone structure (finetuning vs MFT differ
        # in number of energy readout heads → different RNG state at this point).
        if property_head_seed is not None:
            torch.manual_seed(property_head_seed)
        return modules.ScaleShiftMACEProperty(
            backbone=backbone,
            property_head_names=prop_head_names,
            task_dims=task_dims,
            property_intensive=prop_intensive,
            node_feats_scalar_dim=node_feats_scalar_dim,
            node_feats_hidden_dim=node_feats_hidden_dim,
            property_mlp_hidden_dim=getattr(args, "property_mlp_hidden_dim", 240),
            property_mlp_num_layers=getattr(args, "property_mlp_num_layers", 3),
            property_mean=property_mean,
            property_std=property_std,
            property_input_layernorm=getattr(args, "property_input_layernorm", False),
            property_aggregation=prop_aggregation,
            node_feats_vector_dim=node_feats_vector_dim,
        )
    if args.model == "FoundationMACE":
        return modules.ScaleShiftMACE(**model_config_foundation)
    if args.model == "FoundationMACELES":
        from mace.modules.extensions import MACELES

        return MACELES(
            les_arguments=args.les_arguments,
            **model_config_foundation,
        )
    if args.model == "ScaleShiftBOTNet":
        # say it is deprecated
        raise RuntimeError("ScaleShiftBOTNet is deprecated, use MACE instead")
    if args.model == "BOTNet":
        raise RuntimeError("BOTNet is deprecated, use MACE instead")
    if args.model == "AtomicDipolesMACE":
        assert args.loss == "dipole", "Use dipole loss with AtomicDipolesMACE model"
        assert (
            args.error_table == "DipoleRMSE"
        ), "Use error_table DipoleRMSE with AtomicDipolesMACE model"
        return modules.AtomicDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )

    if args.model == "AtomicDielectricMACE":
        args.error_table = "DipolePolarRMSE"
        # std_df = modules.scaling_classes["rms_dipoles_scaling"](train_loader)
        assert (
            args.loss == "dipole_polar"
        ), "Use dipole_polar loss with AtomicDielectricMACE model"
        assert args.error_table in (
            "DipoleRMSE",
            "DipolePolarRMSE",
        ), "Use error_table DipoleRMSE with AtomicDielectricMACE model"
        return modules.AtomicDielectricMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            use_polarizability=True,
        )

    if args.model == "EnergyDipolesMACE":
        assert (
            args.loss == "energy_forces_dipole"
        ), "Use energy_forces_dipole loss with EnergyDipolesMACE model"
        assert (
            args.error_table == "EnergyDipoleRMSE"
        ), "Use error_table EnergyDipoleRMSE with AtomicDipolesMACE model"
        return modules.EnergyDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )
    if args.model == "MACELES":
        from mace.modules.extensions import MACELES

        return MACELES(
            les_arguments=args.les_arguments,
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=[0.0] * len(heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            embedding_specs=args.embedding_specs,
            use_embedding_readout=args.use_embedding_readout,
            use_last_readout_only=args.use_last_readout_only,
            use_agnostic_product=args.use_agnostic_product,
        )
    raise RuntimeError(f"Unknown model: '{args.model}'")
