from robomimic.scripts.config_gen.config_gen_utils import *


def make_generator_helper(args):
    algo_name_short = "bc_xfmr"

    generator = get_generator(
        algo_name="bc",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/bc_transformer.json'),
        args=args,
        algo_name_short=algo_name_short,
    )

    demo_tasks = [
        "PnPCounterToCab",
        "PnPCounterToSink",
        "PnPMicrowaveToCounter",
        "PnPStoveToCounter",
        "OpenSingleDoor",
        "CloseDrawer",
        # "TurnOnMicrowave",
        "TurnOnSinkFaucet",
        "TurnOnStove",
    ]

    all_tasks = [
        "PnPCounterToCab",
        "PnPCabToCounter",
        "PnPCounterToSink",
        "PnPSinkToCounter",
        "PnPCounterToMicrowave",
        "PnPMicrowaveToCounter",
        "PnPCounterToStove",
        "PnPStoveToCounter",
        "OpenSingleDoor",
        "CloseSingleDoor",
        "OpenDoubleDoor",
        "CloseDoubleDoor",
        "OpenDrawer",
        "CloseDrawer",
        "TurnOnSinkFaucet",
        "TurnOffSinkFaucet",
        "TurnSinkSpout",
        "TurnOnStove",
        "TurnOffStove",
        "CoffeeSetupMug",
        "CoffeeServeMug",
        "CoffeePressButton",
        # "TurnOnMicrowave",
        # "TurnOffMicrowave"
    ]

    eval_tasks = [task for task in all_tasks if task not in demo_tasks]

    ### Define dataset variants to train on ###
    generator.add_param(
        key="train.data",
        name="ds",
        group=123456,
        values_and_names=[
            # (get_robocasa_ds("single_stage", src="human", eval=["PnPCounterToSink", "PnPCounterToCab"], filter_key="50_demos"), "human-50"), # training on human datasets
            # (get_robocasa_ds("single_stage", src="mg", eval=["PnPCounterToSink", "PnPCounterToCab"], filter_key="3000_demos"), "mg-3000"), # training on MimicGen datasets
            (get_robocasa_ds(demo_tasks, src="human", eval=[], filter_key="50_demos"), "human-50"), # training on human datasets
            (get_robocasa_ds(demo_tasks, src="mg", eval=[], filter_key="3000_demos"), "mg-3000"), # training on MimicGen datasets
        ]
    )

    """
    ### Uncomment this code to fine-tune on existing checkpoint ###
    generator.add_param(
        key="experiment.ckpt_path",
        name="ckpt",
        group=1389,
        values_and_names=[
            (None, "none"),
            # ("set checkpoint pth path here", "trained-ckpt"),
        ],
    )
    """

    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[get_output_dir(args, algo_dir=algo_name_short)]
    )

    return generator

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()

    make_generator(args, make_generator_helper)
