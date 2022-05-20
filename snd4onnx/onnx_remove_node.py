#! /usr/bin/env python

import os
import sys
from argparse import ArgumentParser
import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.tensor import Variable
from typing import Optional, List

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

OP_TYPES_WITH_AUTOMATIC_ADJUSTMENT_OF_OUTPUT_SHAPE = [
    'Cast',
]

def remove(
    remove_node_names: List[str],
    input_onnx_file_path: Optional[str] = '',
    output_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    non_verbose: Optional[bool] = False,
) -> onnx.ModelProto:

    """
    Parameters
    ----------
    remove_node_names: List[str]
        List of OP names to be deleted.\n\
        e.g. remove_node_names = ['op_name1', 'op_name2', 'op_name3', ...]

    input_onnx_file_path: Optional[str]
        Input onnx file path.\n\
        Either input_onnx_file_path or onnx_graph must be specified.

    output_onnx_file_path: Optional[str]
        Output onnx file path.\n\
        If output_onnx_file_path is not specified, no .onnx file is output.

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n\
        Default: False

    Returns
    -------
    removed_graph: onnx.ModelProto
        OP removed onnx ModelProto.
    """

    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    # Loading Graphs
    # onnx_graph If specified, onnx_graph is processed first
    graph = None
    if not onnx_graph:
        # file existence check
        if not os.path.exists(input_onnx_file_path) or \
            not os.path.isfile(input_onnx_file_path) or \
            not os.path.splitext(input_onnx_file_path)[-1] == '.onnx':
            print(
                f'{Color.RED}ERROR:{Color.RESET} '+
                f'The specified file (.onnx) does not exist. or not an onnx file. File: {input_onnx_file_path}'
            )
            sys.exit(1)
        graph = gs.import_onnx(onnx.load(input_onnx_file_path))
    else:
        graph = gs.import_onnx(onnx_graph)

    remove_nodes = [node for node in graph.nodes if node.name in remove_node_names]

    remove_output_nodes = [graph_output for graph_output in graph.outputs if graph_output.name in remove_node_names]
    if (len(graph.outputs) - len(remove_output_nodes)) <= 0:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            'The number of output_nodes in the graph must be at least 1.'
        )
        sys.exit(1)

    # Minimum number of nodes required is 2 or more
    if len(graph.nodes) < 2:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            'The number of nodes in the graph must be at least 2.'
        )
        sys.exit(1)

    # Minimum number of nodes after deletion is at least 1
    if (len(graph.nodes) - len(remove_nodes)) < 1:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            'At least one node is required for the graph after OP deletion.'
        )
        sys.exit(1)


    with graph.node_ids():
        # Iteration for each node to be deleted
        for rmnode in remove_nodes:
            # Check if it is the first OP of the graph
            rmnode_inputs = []
            matched_graph_input = []
            rmnode_is_first_op = False
            for rmnode_input in rmnode.inputs:
                # Only those with Variable input type are checked
                if isinstance(rmnode_input, Variable):
                    # Check if it matches one of the Inputs in the graph
                    for graph_input in graph.inputs:
                        if rmnode_input == graph_input:
                            rmnode_is_first_op = True
                            matched_graph_input.append(graph_input)
                            rmnode_inputs.append(rmnode_input)
                            break

            # Check if it is the last OP in the graph
            rmnode_outputs = []
            matched_graph_output = []
            rmnode_is_last_op = False
            for rmnode_output in rmnode.outputs:
                # Check only those outputs of type Variable
                if isinstance(rmnode_output, Variable):
                    # Check if it matches any of the Outputs in the graph
                    for graph_output in graph.outputs:
                        if rmnode_output == graph_output:
                            rmnode_is_last_op = True
                            matched_graph_output.append(graph_output)
                            rmnode_outputs.append(rmnode_output)
                            break

            # OPs with two or more input OPs of a graph connected are not allowed to be deleted
            if len(matched_graph_input) >= 2:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} '+
                    'It is not possible to delete an OP to which two or more Input OPs of a graph are connected. '+
                    f'node_name: {rmnode.name}'
                )
                sys.exit(1)
            # If the node to be deleted is responsible for one or more of the final Outputs,
            # delete all of the outputs of the node to be deleted from the final Outputs of the graph.
            # However, at least one final Output of the graph must remain.
            final_graph_output_count = len(graph.outputs)
            remove_outputs = []
            for rmnode_output in rmnode.outputs:
                for graph_output in graph.outputs:
                    if isinstance(rmnode_output, Variable):
                        if rmnode_output == graph_output:
                            final_graph_output_count -= 1
                            remove_outputs.append(rmnode_output)
            # If the node to be deleted is responsible for one or more of the final outputs,
            # delete all the outputs of the node to be deleted from the final output of the graph.
            # However, only when the number of outputs of the OP remains one or more as a result of the deletion,
            # and not when the number of outputs of the OP becomes zero.
            if len(remove_outputs) > 0 and (len(rmnode.outputs) - len(remove_outputs)) >= 1:
                tmp_graph_outputs = []
                for graph_output in graph.outputs:
                    remove_flg = False
                    for remove_output in remove_outputs:
                        if isinstance(rmnode_output, Variable):
                            if graph_output == remove_output:
                                remove_flg = True
                                break
                    if not remove_flg:
                        tmp_graph_outputs.append(graph_output)
                graph.outputs = tmp_graph_outputs
                # Among the outputs of the OP to be deleted,
                # the output information that was employed for the graph output is deleted.
                for remove_output in remove_outputs:
                    rmnode.outputs.remove(remove_output)

            if rmnode_is_first_op:
                # If it is the first OP of the graph,
                # change the Input of the graph to the Input of the next OP of the OP to be deleted.
                # The number of nodes is limited to two or more,
                # so there is always a definite next node

                # If two or more outputs of the OP to be deleted remain that have not been adopted as the final output of the graph,
                # designate all remaining outputs as Inputs of the graph.
                # Identify the next OP based on rmnode.o() of the OP to be deleted,
                # and specify the Input of the graph for the Input of the next OP.
                try:
                    input_change_var_idxs = [idx for idx, input_change_var in enumerate(rmnode.o().inputs) if isinstance(input_change_var, Variable)]
                except:
                    # If it is directly connected to the input of the graph, and yet it is located in the middle,
                    # and yet it is directly connected to the output of the graph,
                    # it is inevitable that an error will occur because the output layer cannot be obtained.
                    # Therefore, this situation, which cannot be checked in advance,
                    # is daringly caught by exception, setting an error message and forcing termination.
                    print(
                        f'{Color.RED}ERROR:{Color.RESET} '+
                        'OPs connected to the input and output of a graph simultaneously cannot be deleted.'
                    )
                    sys.exit(1)
                if len(input_change_var_idxs) < len(rmnode_inputs):
                    # If the number of inputs (Variable) of the OP to be deleted
                    # and the number of inputs (Variable) of the next OP after the deleted OP are different,
                    # the OP cannot be deleted because it cannot be connected.
                    print(
                        f'{Color.RED}ERROR:{Color.RESET} '+
                        'If the number of inputs (Variable) of the OP to be deleted '+
                        'and the number of inputs (Variable) of the next OP after the deleted OP are different, '+
                        'the OP cannot be deleted because it cannot be connected.'
                    )
                    print(
                        f'{Color.RED}ERROR:{Color.RESET} '+
                        f'Remove OP inputs: {len(rmnode_inputs)}, Next OP inputs: {len(input_change_var_idxs)}'
                    )
                    sys.exit(1)
                # Reset what was the input of the OP to be deleted to the next input of the OP to be deleted.
                # However, the order of input cannot be verified at all,
                # because it is set unconditionally from the first to the last.
                for input_change_vars_idx, rmnode_input in zip(input_change_var_idxs, rmnode_inputs):
                    change_output_shape = None
                    next_op_index = None
                    if len(rmnode.o().inputs[input_change_vars_idx].outputs) == 1:
                        # Memorize the modified output shape only for operation types where the next OP
                        # is known to not change the output shape relative to the input shape.
                        if rmnode.o().inputs[input_change_vars_idx].outputs[0].op in OP_TYPES_WITH_AUTOMATIC_ADJUSTMENT_OF_OUTPUT_SHAPE:
                            change_output_shape = rmnode_input.shape
                            next_op_index = rmnode.o().inputs[input_change_vars_idx].outputs[0].id


                    # Check if the input of the OP to be deleted has an OP input other than the input of the graph
                    rmnode_inputs_not_in_graph_inputs = {idx: rmnode_input for idx, rmnode_input in enumerate(rmnode.inputs) if isinstance(rmnode_input, Variable)}
                    if len(rmnode_inputs_not_in_graph_inputs) >= 1:
                        # Set the output of the OP before the OP to be deleted to the input of the OP following the OP to be deleted.
                        # Force the input with the smallest sequential number among multiple inputs.
                        for idx, rmnode_input in enumerate(rmnode.o().inputs):
                            for rmnode_output in rmnode.outputs:
                                if rmnode_input == rmnode_output:
                                    rmnode.o().inputs[idx] = list(rmnode_inputs_not_in_graph_inputs.values())[0]
                                    break
                            else:
                                continue
                            break

                        if not non_verbose:
                            print(
                                f'{Color.YELLOW}WARNING:{Color.RESET} '+
                                'There may be a mismatch in the input/output shapes '+
                                'before and after the OP to be deleted. Check the graph carefully.'
                            )

                        # Forces the shape of the Output of the next OP to fit the input shape of the OP to be deleted.
                        if change_output_shape is not None:
                            for output in graph.nodes[next_op_index].outputs:
                                if isinstance(output, Variable):
                                    output.shape = change_output_shape
                                    break

                # Clear all output of OPs to be deleted
                rmnode.outputs.clear()

            if rmnode_is_last_op:
                # If it was the last OP in the graph
                # Reassign the output specified as Input for the OP to be deleted to the output of the graph.

                # Deletes the Output of the OP to be deleted from the Output of the graph.
                for remove_output in matched_graph_output:
                    if remove_output in graph.outputs:
                        graph.outputs.remove(remove_output)
                # Add the Input of the OP to be deleted to the Output of the graph.
                for rmnode_input in rmnode.inputs:
                    graph.outputs.append(rmnode_input)
                # Clear all output of OPs to be deleted
                rmnode.outputs.clear()

            if not rmnode_is_first_op and not rmnode_is_last_op:
                # If it was neither the first nor the last OP
                # Re-designate what was designated as Input of the OP to be deleted as Input of the next OP.
                # However, if there is a gap between the number of inputs in the OP to be deleted
                # and the number of inputs in the next OP, an error occurs.
                inp_node = rmnode.i()
                out_node = rmnode.o()

                output_change_var_idxs = [idx for idx, output_change_var in enumerate(inp_node.outputs) if isinstance(output_change_var, Variable)]
                input_change_var_idxs = [idx for idx, input_change_var in enumerate(out_node.inputs) if isinstance(input_change_var, Variable)]

                if len(output_change_var_idxs) != len(input_change_var_idxs):
                    print(
                        f'{Color.RED}ERROR:{Color.RESET} '+
                        'If the number of outputs of the OP immediately before the OP to be deleted '+
                        'is different from the number of inputs of the OP immediately after the OP to be deleted, '+
                        'the OP cannot be automatically reconnected.'
                    )
                    print(
                        f'{Color.RED}ERROR:{Color.RESET} '+
                        f'Remove OP inputs: {len(rmnode_inputs)}, Next OP inputs: {len(input_change_var_idxs)}'
                    )
                    sys.exit(1)

                for output_change_var_idx, input_change_var_idx in zip(output_change_var_idxs, input_change_var_idxs):
                    rmnode.i().outputs[output_change_var_idx] = rmnode.o().inputs[input_change_var_idx]

                # Clear all output of OPs to be deleted
                rmnode.outputs.clear()

            if rmnode_is_first_op and rmnode_is_last_op:
                # If it is both the first and the last OP,
                # it is considered too complicated to process and is warn for the time being.
                if not non_verbose:
                    print(
                        f'{Color.YELLOW}WARNING:{Color.RESET} '+
                        'Since the OP to be deleted is both the beginning and the end of the graph, '+
                        'it is treated as unprocessable at this time. '+
                        'Carefully check the geometry of the generated model.'
                    )

    graph.cleanup().toposort()

    # Delete any unused graph inputs
    remove_graph_inputs = []
    for graph_input in graph.inputs:
        graph_unused_input = True
        for node in graph.nodes:
            for node_input in node.inputs:
                if graph_input == node_input:
                    graph_unused_input = False
                    break
            else:
                continue
            break
        if graph_unused_input:
            remove_graph_inputs.append(graph_input)
    for remove_graph_input in remove_graph_inputs:
        graph.inputs.remove(remove_graph_input)
    graph.cleanup().toposort()

    # Delete output nodes
    graph.outputs = [graph_output for graph_output in graph.outputs if graph_output.name not in remove_node_names]

    graph.cleanup().toposort()

    new_model = None
    try:
        new_model = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
    except:
        new_model = gs.export_onnx(graph)
        print(
            f'{Color.YELLOW}WARNING:{Color.RESET} '+
            'The input shape of the next OP does not match the output shape. '+
            'Be sure to open the .onnx file to verify the certainty of the geometry.'
        )

    # Save
    if output_onnx_file_path:
        onnx.save(new_model, f'{output_onnx_file_path}')

    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} Finish!')

    return new_model


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--remove_node_names',
        type=str,
        required=True,
        nargs='+',
        help='ONNX node name to be deleted.'
    )
    parser.add_argument(
        '--input_onnx_file_path',
        type=str,
        required=True,
        help='Input onnx file path.'
    )
    parser.add_argument(
        '--output_onnx_file_path',
        type=str,
        required=True,
        help='Output onnx file path.'
    )
    parser.add_argument(
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    remove_node_names = args.remove_node_names
    input_onnx_file_path = args.input_onnx_file_path
    output_onnx_file_path = args.output_onnx_file_path
    non_verbose = args.non_verbose

    onnx_graph = remove(
        remove_node_names=remove_node_names,
        input_onnx_file_path=input_onnx_file_path,
        output_onnx_file_path=output_onnx_file_path,
        non_verbose=non_verbose,
    )


if __name__ == '__main__':
    main()