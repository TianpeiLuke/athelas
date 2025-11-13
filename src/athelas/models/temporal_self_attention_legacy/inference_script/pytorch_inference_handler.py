import os
import json
import torch
import pickle
import numpy as np

from scipy.interpolate import BSpline

from CategoricalTransformer import CategoricalTransformer
from params import (
    seq_len,
    SEP,
    seq_cat_vars,
    seq_num_vars,
    dense_num_vars,
    input_data_seq_cat_vars,
    input_data_seq_num_vars,
    input_data_dense_num_vars,
    input_data_seq_cat_otf_vars,
    input_data_seq_num_otf_vars,
    numerical_cat_vars,
    numerical_cat_vars_indices,
)
from models import OrderFeatureAttentionClassifier

TRAINED_LOCAL_RN_FILENAME = "isSmall_0_OrderFeature.pt"

mtx_from_dict_fill_default = (
    lambda input_data, var_list_otf, var_list, map_dict: np.array(
        [
            [
                map_dict[var_list[i]] if a in ["", "My Text String"] else a
                for a in input_data[var_list_otf[i]].split(SEP)
            ]
            for i in range(len(var_list_otf))
        ]
    ).transpose()
)
arr_from_dict_fill_default = lambda input_data, var_list, map_dict: np.expand_dims(
    np.array(
        [
            map_dict[var_list[i]]
            if input_data[var_list[i]] in ["", "My Text String"]
            else input_data[var_list[i]]
            for i in range(len(var_list))
        ]
    ),
    axis=0,
)

mtx_from_dict_fill_default_w_min_len = (
    lambda input_data, var_list_otf, var_list, map_dict, min_len: np.array(
        [
            [
                map_dict[var_list[i]] if a in ["", "My Text String"] else a
                for a in input_data[var_list_otf[i]].split(SEP)[-min_len:]
            ]
            for i in range(len(var_list_otf))
        ]
    ).transpose()
)

percentile_score_map = None
seq_num_scale_ = None
seq_num_min_ = None
num_static_scale_ = None
num_static_min_ = None
params_probability = None
categorical_map = None
default_value_dict = None
spline = lambda x: x


def input_fn(request_body, content_type):
    """
    Process the input request of the sagemaker endpoint, we can add our preprocessing logic at here. Its return value is the argument input_data for function predict_fn
    :param request_body: nput request of the sagemaker endpoint
    :param content_type: content type
    :return: json parsed data
    """

    if content_type == "application/json":
        # Read the raw input data as JSON.
        return json.loads(request_body)
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(prediction, accept):
    if accept == "application/json":
        raw_score = prediction
        percentile_score = get_percentile_score(raw_score, percentile_score_map)
        probability_score = get_probability_score(raw_score)

        if "e" in str(raw_score):
            raw_score = np.format_float_positional(raw_score)
        if "e" in str(percentile_score):
            percentile_score = np.format_float_positional(percentile_score)
        if "e" in str(probability_score):
            probability_score = np.format_float_positional(probability_score)

        instances = [
            {
                "score-percentile": str(percentile_score),
                "legacy-score": str(raw_score),
                "ProbabilityScore": str(probability_score),
            }
        ]

        #         print('Check output. '
        #                   'Source: {}'.format(instances))

        return {"predictions": instances}

    else:
        raise ValueError("output format doesn't support {}".format(accept))


def predict_fn(input_data, model):
    """
    Process input data and return predictions
    :param input_data: input data
    :param model: model
    :return: prediction
    """

    ret, seq_cat_mtx, seq_num_mtx, dense_num_arr = data_parsing(input_data)

    if ret:
        key_padding_mask = torch.logical_not(
            torch.nn.functional.pad(
                torch.Tensor(seq_num_mtx[:, -1]), (0, 1), value=True
            )
        )
        key_padding_mask = key_padding_mask.unsqueeze(0)

        with torch.no_grad():
            raw_score, _ = model.forward(
                x_cat=torch.Tensor(seq_cat_mtx).unsqueeze(dim=0),
                x_num=torch.Tensor(seq_num_mtx[:, :-2]).unsqueeze(dim=0),
                x_engineered=torch.Tensor(dense_num_arr),
                time_seq=torch.Tensor(seq_num_mtx[:, -2])
                .unsqueeze(dim=-1)
                .unsqueeze(dim=0),
                attn_mask=None,
                key_padding_mask=key_padding_mask,
            )
            model_score = torch.softmax(raw_score[0], dim=0)[1]
        model_score = model_score.detach().cpu().tolist()

    else:
        print("Sanity check failed. Model return -1")
        model_score = -1

    return model_score


#     model_output = {'tsa_model_score': model_score}
#     output_data = {'predictions': [model_output]}  # require format by AMES

#     return output_data


def model_fn(model_dir):
    """
    Deserialize fitted model
    :param model_dir: model dir
    :return: serialized fitted model
    """

    global \
        percentile_score_map, \
        seq_num_scale_, \
        seq_num_min_, \
        num_static_scale_, \
        num_static_min_, \
        params_probability, \
        categorical_map, \
        default_value_dict, \
        spline

    preprocessor_file = "preprocessor.pkl"
    preprocessor = pickle.load(open(os.path.join(model_dir, preprocessor_file), "rb"))
    percentile_file = "percentile_score.pkl"
    percentile_score_map = pickle.load(
        open(os.path.join(model_dir, percentile_file), "rb")
    )
    #     percentile_score_map = preprocessor['bin_map']
    seq_num_scale_ = preprocessor["seq_num_scale_"]
    seq_num_min_ = preprocessor["seq_num_min_"]
    num_static_scale_ = preprocessor["num_static_scale_"]
    num_static_min_ = preprocessor["num_static_min_"]

    # Will be deleted when generating new min-max scaler parameters
    num_static_scale_ = np.delete(num_static_scale_, [266, 267])
    num_static_min_ = np.delete(num_static_min_, [266, 267])

    bspline_file = "bspline_parameters.json"
    with open(os.path.join(model_dir, bspline_file), "r") as f:
        params_probability = json.load(f)
        spline = BSpline(
            t=params_probability["knots"],
            c=params_probability["coefficients"],
            k=params_probability["degree"],
            extrapolate=False,
        )

    cat_to_index_file = "cat_to_index.json"
    with open(os.path.join(model_dir, cat_to_index_file), "r") as f:
        categorical_map = json.load(f)

    default_value_dict_file = "default_value_dict.json"
    with open(os.path.join(model_dir, default_value_dict_file), "r") as f:
        default_value_dict = json.load(f)

    return load_model(model_dir)


def load_model(model_dir):
    """
    Load trained pytorch model
    :param model_dir: model director
    :return: loaded pytorch model
    """

    n_cat_features = 108
    n_num_features = 51
    n_classes = 2
    n_embedding = 2403
    seq_len = 51
    n_engineered_num_features = 413
    dim_embedding_table = 64
    dim_attn_feedforward = 256
    use_mlp = 0
    num_heads = 1
    dropout = 0.1
    n_layers_order = 6
    n_layers_feature = 4
    emb_tbl_use_bias = 1
    use_moe = 0
    num_experts = 5
    use_time_seq = 1

    # create the model
    print("Loading trained model")
    model = OrderFeatureAttentionClassifier(
        n_cat_features,
        n_num_features,
        n_classes,
        n_embedding,
        seq_len,
        n_engineered_num_features,
        dim_embedding_table,
        dim_attn_feedforward,
        use_mlp,
        num_heads,
        dropout,
        n_layers_order,
        n_layers_feature,
        emb_tbl_use_bias,
        use_moe,
        num_experts,
        use_time_seq,
    )

    # load trained model state
    model_state_fp = os.path.join(model_dir, TRAINED_LOCAL_RN_FILENAME)
    model_state = torch.load(model_state_fp, map_location="cpu")
    model.load_state_dict(model_state)
    model.eval()

    torch.set_num_threads(1)

    print("Finished Loading trained model")

    return model


def data_parsing(input_data):
    """
    Data parsing and sanity check
    :param input_data: input data
    :return: pass_check, terminal_vec_lst, neighbor_vec_lst, neighbor_y_lst
    """
    global seq_num_scale_, seq_num_min_, num_static_scale_, num_static_min_, categorical_map, default_value_dict

    #     print('Check input. '
    #                   'Source: {}'.format(input_data))

    if not isinstance(input_data, dict):
        print(
            "Sanity check failed. Input data is not a dict obj. Source: {}".format(
                input_data
            )
        )
        return False, None, None, None

    input_data["objectId"] = "CURRENT"

    for VAR in input_data_seq_cat_otf_vars:
        if VAR not in input_data:
            print(
                "Sanity check failed. "
                "Input data does not contain required key in input_data_seq_cat_otf_vars. "
                "Source: {}".format(input_data)
            )
            return False, None, None, None

    for VAR in input_data_seq_cat_vars:
        if VAR not in input_data:
            print(
                "Sanity check failed. "
                "Input data does not contain required key in input_data_seq_cat_vars. "
                "Source: {}".format(input_data)
            )
            return False, None, None, None

    for VAR in input_data_seq_num_otf_vars:
        if VAR not in input_data:
            print(
                "Sanity check failed. "
                "Input data does not contain required key in input_data_seq_num_otf_vars. "
                "Source: {}".format(input_data)
            )
            return False, None, None, None

    for VAR in input_data_seq_num_vars:
        if VAR not in input_data:
            print(
                "Sanity check failed. "
                "Input data does not contain required key in input_data_seq_num_vars. "
                "Source: {}".format(input_data)
            )
            return False, None, None, None

    for VAR in input_data_dense_num_vars:
        if VAR not in input_data:
            print(
                "Sanity check failed. "
                "Input data does not contain required key in input_data_dense_num_vars. "
                "Source: {}".format(input_data)
            )
            return False, None, None, None

    no_history_flag = (
        input_data[
            "payment_risk.bfs_order_cat_seq_by_cid.c_billingaddrlatlongconfidence_seq"
        ]
        in ["", "My Text String"]
    ) or (
        input_data["payment_risk.bfs_order_cat_seq_by_cid.c_declr_seq"]
        in ["", "My Text String"]
    )

    otf_len_1 = len(
        input_data["payment_risk.bfs_order_cat_seq_by_cid.c_declr_seq"].split(SEP)
    )
    otf_len_2 = len(
        input_data[
            "payment_risk.bfs_order_cat_seq_by_cid.c_billingaddrlatlongconfidence_seq"
        ].split(SEP)
    )
    min_len = min(otf_len_1, otf_len_2)
    if otf_len_1 != otf_len_2:
        print(
            "Sanity check warning. "
            "Input data OTF not matured. "
            "Lengths of two batches not matched. "
            "Source: {}".format(input_data)
        )
        otf_history_mismatch_flag = True
    #         return False, None, None, None
    else:
        otf_history_mismatch_flag = False

    for i in numerical_cat_vars_indices:
        #         if cur_var in ['','My Text String']:
        #             print('Sanity check failed. '
        #                   'Input data numeric categorical variable value wrong. '
        #                   'Source: {}'.format(input_data))
        #             return False, None, None, None

        cur_var = input_data[input_data_seq_cat_vars[i]]
        if cur_var not in ["", "My Text String", "false"]:
            cur_var = str(int(float(cur_var)))
            input_data[input_data_seq_cat_vars[i]] = cur_var

        if not no_history_flag:
            var_seq_list = [
                str(int(float(var_))) if var_ != "" else var_
                for var_ in input_data[input_data_seq_cat_otf_vars[i]].split(SEP)
            ]
            input_data[input_data_seq_cat_otf_vars[i]] = SEP.join(var_seq_list)

    columns_list = input_data_seq_cat_vars[:-2]
    transform_object = CategoricalTransformer(
        categorical_map=categorical_map, columns_list=columns_list
    )

    # parse string to lst
    if not no_history_flag:
        if otf_history_mismatch_flag:
            seq_cat_vars_mtx = mtx_from_dict_fill_default_w_min_len(
                input_data,
                input_data_seq_cat_otf_vars,
                input_data_seq_cat_vars,
                default_value_dict,
                min_len,
            )
            seq_num_vars_mtx = mtx_from_dict_fill_default_w_min_len(
                input_data,
                input_data_seq_num_otf_vars,
                input_data_seq_num_vars,
                default_value_dict,
                min_len,
            )
        else:
            seq_cat_vars_mtx = mtx_from_dict_fill_default(
                input_data,
                input_data_seq_cat_otf_vars,
                input_data_seq_cat_vars,
                default_value_dict,
            )
            seq_num_vars_mtx = mtx_from_dict_fill_default(
                input_data,
                input_data_seq_num_otf_vars,
                input_data_seq_num_vars,
                default_value_dict,
            )
    seq_cat_vars_lst = arr_from_dict_fill_default(
        input_data, input_data_seq_cat_vars, default_value_dict
    )
    seq_num_vars_lst = arr_from_dict_fill_default(
        input_data, input_data_seq_num_vars, default_value_dict
    )
    dense_num_vars_lst = arr_from_dict_fill_default(
        input_data, input_data_dense_num_vars, default_value_dict
    )

    if not no_history_flag:
        if len(seq_cat_vars_mtx) == len(seq_num_vars_mtx):
            if sum(
                seq_cat_vars_mtx[:, -1].argsort() == seq_cat_vars_mtx[:, -1].argsort()
            ) != len(seq_cat_vars_mtx):
                print(
                    "Sanity check warning. "
                    "Input data OTFs have same length but mismatch lines. "
                    "Use only currently order for evaluation. "
                    "Source: {} {}".format(seq_cat_vars_mtx, seq_num_vars_mtx)
                )
                #                 return False, None, None, None
                #                 no_history_flag = True
                intersect1d, comm1, comm2 = np.intersect1d(
                    seq_cat_vars_mtx[:, -1],
                    seq_num_vars_mtx[:, -1],
                    return_indices=True,
                )
                seq_cat_vars_mtx = seq_cat_vars_mtx[comm1, :]
                seq_num_vars_mtx = seq_num_vars_mtx[comm2, :]
        else:
            print(
                "Sanity check warning. "
                "Input data OTFs have mismatch length. "
                "Use only currently order for evaluation. "
                "Source: {} {}".format(seq_cat_vars_mtx, seq_num_vars_mtx)
            )
            #             no_history_flag = True
            #             return False, None, None, None
            intersect1d, comm1, comm2 = np.intersect1d(
                seq_cat_vars_mtx[:, -1], seq_num_vars_mtx[:, -1], return_indices=True
            )
            seq_cat_vars_mtx = seq_cat_vars_mtx[comm1, :]
            seq_num_vars_mtx = seq_num_vars_mtx[comm2, :]

    if not no_history_flag:
        seq_cat_mtx = np.concatenate(
            [seq_cat_vars_mtx[:, :-2], seq_cat_vars_lst[:, :-2]]
        )
    else:
        seq_cat_mtx = seq_cat_vars_lst[:, :-2]
    #     seq_cat_mtx[seq_cat_mtx=='']='-1'
    seq_cat_mtx = seq_cat_mtx.astype("str")
    seq_cat_mtx = transform_object.transform(seq_cat_mtx)
    seq_cat_mtx[seq_cat_mtx == "None"] = "0"
    seq_cat_mtx = seq_cat_mtx.astype(int)
    if not no_history_flag:
        seq_cat_mtx = np.pad(
            seq_cat_mtx, [(seq_len - 1 - len(seq_cat_vars_mtx), 0), (0, 0)]
        )
    else:
        seq_cat_mtx = np.pad(seq_cat_mtx, [(seq_len - 1, 0), (0, 0)])

    if not no_history_flag:
        seq_num_mtx = np.concatenate(
            [seq_num_vars_mtx[:, :-1], seq_num_vars_lst[:, :-1]]
        )
    else:
        seq_num_mtx = seq_num_vars_lst[:, :-1]
    seq_num_mtx = np.concatenate(
        [seq_num_mtx, np.ones((seq_num_mtx.shape[0], 1))], axis=1
    )
    seq_num_mtx = seq_num_mtx.astype(float)
    seq_num_mtx[:, :-2] = seq_num_mtx[:, :-2] * np.array(seq_num_scale_) + np.array(
        seq_num_min_
    )
    seq_num_mtx[:, -2] = seq_num_mtx[-1, -2] - seq_num_mtx[:, -2]
    if np.max(seq_num_mtx[:, -2]) > 10000000:
        print(
            "Sanity check failed. "
            "Sequence OTF recorded wrong orderDate. "
            "Source: {}".format(input_data)
        )
        return False, None, None, None

    if not no_history_flag:
        seq_num_mtx = np.pad(
            seq_num_mtx, [(seq_len - 1 - len(seq_num_vars_mtx), 0), (0, 0)]
        )
    else:
        seq_num_mtx = np.pad(seq_num_mtx, [(seq_len - 1, 0), (0, 0)])

    dense_num_vars_lst = dense_num_vars_lst[:, :-2]
    dense_num_vars_lst = dense_num_vars_lst.astype(float)
    dense_num_arr = dense_num_vars_lst * np.array(num_static_scale_) + np.array(
        num_static_min_
    )

    print("Data sanity check passed and preprocessed.")
    return True, seq_cat_mtx, seq_num_mtx, dense_num_arr


def get_probability_score(raw_score):
    global spline
    if raw_score <= 0:
        return 0
    if raw_score >= 1:
        return 1

    linear_prediction = spline(raw_score)
    if linear_prediction > 25:
        return 1
    tmp = np.exp(linear_prediction)
    probability_score = tmp / (tmp + 1)

    return probability_score


def binary_search(target, percentile_score):
    start = 0
    end = len(percentile_score) - 1
    while start <= end:
        mid = (start + end) // 2
        if percentile_score[mid][0] == target:
            return mid
        elif percentile_score[mid][0] < target:
            start = mid + 1
        else:
            end = mid - 1
    return end


def get_percentile_score(raw_score, percentile_score):
    if raw_score <= 0:
        return 0
    if raw_score >= 1:
        return 1
    index = binary_search(raw_score, percentile_score)
    if index == len(percentile_score) - 1:
        return percentile_score[index][1]
    return percentile_score[index + 1][1]
