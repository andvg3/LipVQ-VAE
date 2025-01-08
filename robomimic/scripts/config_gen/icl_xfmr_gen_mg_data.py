from robomimic.scripts.config_gen.config_gen_utils import *


def make_generator_helper(args):
    algo_name_short = "icl_xfmr_{}".format(args.task_name)

    generator = get_generator(
        algo_name="icl",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/icl_transformer.json'),
        args=args,
        algo_name_short=algo_name_short,
    )

    values_and_names = []
    for task in args.task_list:
        values_and_names.append((get_robocasa_ds(task, src="mg", eval=[], filter_key="3000_demos"), "mg-3000"))

    ### Define dataset variants to train on ###
    generator.add_param(
        key="train.data",
        name="ds",
        group=123456,
        values_and_names=values_and_names
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

    tasks = {
        "PnP": [
            "PnPCounterToCab",
            "PnPCabToCounter",
            "PnPCounterToSink",
            "PnPSinkToCounter",
            "PnPCounterToMicrowave",
            "PnPMicrowaveToCounter",
            "PnPCounterToStove",
            "PnPStoveToCounter"
        ],
        "OpenCloseDoors": [
            "OpenSingleDoor",
            "CloseSingleDoor",
            "OpenDoubleDoor",
            "CloseDoubleDoor"
        ],
        "OpenCloseDrawers": [
            "OpenDrawer",
            "CloseDrawer"
        ],
        "TurnLevers": [
            "TurnOnSinkFaucet",
            "TurnOffSinkFaucet",
            "TurnSinkSpout"
        ],
        "TwistKnobs": [
            "TurnOnStove",
            "TurnOffStove"
        ],
        "Insertion": [
            "CoffeeSetupMug",
            "CoffeeServeMug"
        ],
        "PressButtons": [
            "CoffeePressButton",
            "TurnOnMicrowave",
            "TurnOffMicrowave"
        ]
    }

    args.task_name = None
    for task_name in tasks:
        args.task_name = task_name 
        args.task_list = tasks[task_name]
        make_generator(args, make_generator_helper)
