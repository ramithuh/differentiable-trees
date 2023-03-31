import pprint
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a differentiable tree model. \n sample usage :  python train.py -l 8 -m 10 -sl 50 -fs')

    parser.add_argument('-gt', '--groundtruth', action='store_true', help='Retrieve groundtruth?')
    parser.set_defaults(groundtruth=True)


    parser.add_argument('-n','--notes'       , help='notes'            , required=True, type = str, default = '')
    #tree and seq generate parameters
    parser.add_argument('-l','--leaves'      , help='# of leaves?'    , required=False, type = int,  default = 4)
    parser.add_argument('-m','--mutations'   , help='# of leaves?'    , required=False, type = int,  default = 10)
    parser.add_argument('-sl','--seq_length' , help='length of seq'   , required=False, type = int,  default = 50)
    parser.add_argument('-nl','--letters'    , help='# of letters?'    , required=False, type = int, default = 20)
    parser.add_argument('-s','--seed'        , help='seed'            , required=False, type = int)

    parser.add_argument('-ai','--alternate_interval', help='alternate_interval', required=False, type = int)
    parser.add_argument('-tLs','--tree_loss_schedule' , help='tree loss schedule', required=False, type = str)
    parser.add_argument('-lr','--learning_rate'      , help='learning rate'    , required=False, type = float)

    ## what to train?
    parser.add_argument('-fs', '--fix_seqs', action='store_true', help='Fix sequences when training ?')
    parser.set_defaults(fix_seqs=False)

    parser.add_argument('-ft', '--fix_tree', action='store_true', help='Fix tree when training ?')
    parser.set_defaults(fix_tree=False)

    parser.add_argument('-it', '--initialize_tree', action='store_true', help='Initialize Tree with groundtruth?')
    parser.set_defaults(initialize_tree=False)

    parser.add_argument('-is', '--initialize_seq', action='store_true', help='Initialize Seq with groundtruth?')
    parser.set_defaults(initialize_seq=False)

    parser.add_argument('-alt', '--alternate_optimization', action='store_true', help='Alternate tree and seq optimization?')
    parser.set_defaults(alternate_optimization=False)

    return parser.parse_args()

def pretty_print_dict(d):
    # Format the dictionary using the pprint module
    
    formatted_dict = pprint.pformat(d, width=1)
    
    # Return the formatted dictionary as a string
    return formatted_dict
