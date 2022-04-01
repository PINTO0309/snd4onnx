#! /usr/bin/env python

import os
import sys
import shutil
from argparse import ArgumentParser
import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.tensor import Variable

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

def main():
    parser = ArgumentParser()
    parser.add_argument('onnx_file_path', type=str, help='Input onnx file path.')
    parser.add_argument('remove_node_names', type=str, help='ONNX node name to be deleted. Comma delimited.')
    args = parser.parse_args()


    work_file_path = shutil.copy(args.onnx_file_path, f'{os.path.splitext(args.onnx_file_path)[0]}_removed.onnx')

    graph = gs.import_onnx(onnx.load(work_file_path))
    remove_node_names = args.remove_node_names.split(',')
    remove_nodes = [node for node in graph.nodes if node.name in remove_node_names]

    # 必要最低限のノード数は２以上
    if len(graph.nodes) < 2:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            'The number of nodes in the graph must be at least 2.'
        )
        sys.exit(0)

    # 削除後の最低限のノード数は１以上
    if (len(graph.nodes) - len(remove_nodes)) < 1:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            'At least one node is required for the graph after OP deletion.'
        )
        sys.exit(0)


    with graph.node_ids():
        # 削除ノードごとに繰り返し処理
        for rmnode in remove_nodes:
            # グラフの最初のOPかどうかをチェック
            rmnode_inputs = []
            matched_graph_input = []
            rmnode_is_first_op = False
            for rmnode_input in rmnode.inputs:
                # インプットの型がVariableのもののみをチェック対象にする
                if isinstance(rmnode_input, Variable):
                    # グラフのいずれかのInputと一致するかどうかをチェック
                    for graph_input in graph.inputs:
                        if rmnode_input == graph_input:
                            rmnode_is_first_op = True
                            matched_graph_input.append(graph_input)
                            rmnode_inputs.append(rmnode_input)
                            break

            # グラフの最後のOPかどうかをチェック
            rmnode_outputs = []
            matched_graph_output = []
            rmnode_is_last_op = False
            for rmnode_output in rmnode.outputs:
                # アウトプットの型がVariableのもののみをチェック対象にする
                if isinstance(rmnode_output, Variable):
                    # グラフのいずれかのOutputと一致するかどうかをチェック
                    for graph_output in graph.outputs:
                        if rmnode_output == graph_output:
                            rmnode_is_last_op = True
                            matched_graph_output.append(graph_output)
                            rmnode_outputs.append(rmnode_output)
                            break

            # グラフの入力OPが２個以上連結されたOPは削除不可とする
            if len(matched_graph_input) >= 2:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} '+
                    'It is not possible to delete an OP to which two or more Input OPs of a graph are connected. '+
                    f'node_name: {rmnode.name}'
                )
                sys.exit(0)
            # 削除対象のノードが最終Outputの１つ以上を担っていた場合、グラフの最終Outputから削除対象ノードの出力分だけを全部削除する。ただし、グラフの最終Outputが１個以上残ること
            final_graph_output_count = len(graph.outputs)
            remove_outputs = []
            for rmnode_output in rmnode.outputs:
                for graph_output in graph.outputs:
                    if isinstance(rmnode_output, Variable):
                        if rmnode_output == graph_output:
                            final_graph_output_count -= 1
                            remove_outputs.append(rmnode_output)
            # 削除対象のノードが最終Outputの１つ以上を担っていた場合、グラフの最終Outputから削除対象ノードの出力分だけを全部削除
            # ただし、削除した結果、OPの出力数が１個以上残る場合のみとし、OPの出力数がゼロ個になる場合は削除しない
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
                # 削除対象OPの出力のうち、グラフ出力に採用されていた出力情報を抹消する
                for remove_output in remove_outputs:
                    rmnode.outputs.remove(remove_output)

            if rmnode_is_first_op:
                # グラフの最初のOPだった場合はグラフのInputを削除するOPの次のOPのInputに変更する
                # ノードの数は２個以上に限定されているので必ず次のノードがあることが確定している

                # 削除対象OPの出力のうち、グラフの最終出力に採用されていなかった出力の数が２個以上残っている場合は、残っている出力を全てグラフのInputに指定する
                # 削除対象のOPのrmnode.o()をもとにして次のOPを特定し、次のOPのInputにグラフのInputを指定する
                try:
                    input_change_var_idxs = [idx for idx, input_change_var in enumerate(rmnode.o().inputs) if isinstance(input_change_var, Variable)]
                except:
                    # グラフの入力に直結されていて、なおかつ中間に位置し、なおかつグラフの出力に直結されている場合は出力レイヤーが取得できずにエラーになることが避けられない
                    # したがって、事前チェック不可能なこの状況をあえて例外で捕捉し、エラーメッセージを設定して強制終了する
                    print(
                        f'{Color.RED}ERROR:{Color.RESET} '+
                        'OPs connected to the input and output of a graph simultaneously cannot be deleted.'
                    )
                    sys.exit(0)
                if len(input_change_var_idxs) < len(rmnode_inputs):
                    # 削除するOPの入力数(Variable数)と削除OPの次のOPの入力数(Variable数)が異なる場合は繋げられないので削除不可
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
                    sys.exit(0)
                # 削除対象OPの入力だったものを削除対象OPの次の入力に再設定
                # ただし、無条件に先頭から順番にセットしていくので入力の順序の確からしさは一切検証できない
                for input_change_vars_idx, rmnode_input in zip(input_change_var_idxs, rmnode_inputs):
                    change_output_shape = None
                    next_op_index = None
                    if len(rmnode.o().inputs[input_change_vars_idx].outputs) == 1:
                        # 次のOPが入力形状に対して出力形状が変わらないことが分かっているオペレーションタイプのみ変更後の出力形状を記憶する
                        if rmnode.o().inputs[input_change_vars_idx].outputs[0].op in OP_TYPES_WITH_AUTOMATIC_ADJUSTMENT_OF_OUTPUT_SHAPE:
                            change_output_shape = rmnode_input.shape
                            next_op_index = rmnode.o().inputs[input_change_vars_idx].outputs[0].id


                    # 削除対象OPの入力にグラフの入力以外のOPの入力があるかどうかをチェック
                    rmnode_inputs_not_in_graph_inputs = {idx: rmnode_input for idx, rmnode_input in enumerate(rmnode.inputs) if isinstance(rmnode_input, Variable)}
                    if len(rmnode_inputs_not_in_graph_inputs) >= 1:
                        # 削除対象OPの次のOPの入力に削除対象OPの前のOPの出力を設定する
                        # 複数ある入力のうち、いちばん連番が小さい入力を強制的に採用する
                        for idx, rmnode_input in enumerate(rmnode.o().inputs):
                            for rmnode_output in rmnode.outputs:
                                if rmnode_input == rmnode_output:
                                    rmnode.o().inputs[idx] = list(rmnode_inputs_not_in_graph_inputs.values())[0]
                                    break
                            else:
                                continue
                            break

                        print(
                            f'{Color.YELLOW}WARNING:{Color.RESET} '+
                            'There may be a mismatch in the input/output shapes '+
                            'before and after the OP to be deleted. Check the graph carefully.'
                        )

                        # 次のOPのOutputの形状を強制的に削除対象OPの入力形状にフィットさせる
                        if change_output_shape is not None:
                            for output in graph.nodes[next_op_index].outputs:
                                if isinstance(output, Variable):
                                    output.shape = change_output_shape
                                    break

                # 削除対象OPの出力をすべてクリア
                rmnode.outputs.clear()

            if rmnode_is_last_op:
                # グラフの最後のOPだった場合
                # 削除対象OPのInputに指定されていた出力をグラフの出力に再設定する

                # 削除対象OPのOutputをグラフのOutputから削除
                for remove_output in matched_graph_output:
                    if remove_output in graph.outputs:
                        graph.outputs.remove(remove_output)
                # 削除対象OPのInputをグラフのOutputに追加
                for rmnode_input in rmnode.inputs:
                    graph.outputs.append(rmnode_input)
                # 削除対象OPの出力をすべてクリア
                rmnode.outputs.clear()

            if not rmnode_is_first_op and not rmnode_is_last_op:
                # 最初のOPでも最後のOPでもなかった場合
                # 削除対象OPのInputに指定されていたものを次のOPのInputに再指定する
                # ただし、削除対象OPのInputの数と次のOPのInputの数に乖離があるときはエラーとする
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
                        'Remove OP inputs: {len(rmnode_inputs)}, Next OP inputs: {len(input_change_var_idxs)}'
                    )
                    sys.exit(0)

                for output_change_var_idx, input_change_var_idx in zip(output_change_var_idxs, input_change_var_idxs):
                    rmnode.i().outputs[output_change_var_idx] = rmnode.o().inputs[input_change_var_idx]

                # 削除対象OPの出力をすべてクリア
                rmnode.outputs.clear()

            if rmnode_is_first_op and rmnode_is_last_op:
                # 最初のOPでもあり最後のOPでもあった場合は複雑なので処理不能として当面はワーニングにする
                print(
                    f'{Color.YELLOW}WARNING:{Color.RESET} '+
                    'Since the OP to be deleted is both the beginning and the end of the graph, '+
                    'it is treated as unprocessable at this time. '+
                    'Carefully check the geometry of the generated model.'
                )

    graph.cleanup().toposort()

    # 未使用となったグラフのInputがあれば削除
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
    onnx.save(new_model, f'{work_file_path}')

if __name__ == '__main__':
    main()