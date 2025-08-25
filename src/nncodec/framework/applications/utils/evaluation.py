'''
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or 
other Intellectual Property Rights other than the copyrights concerning 
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2019-2025, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The NNCodec Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

import torch
import numpy as np
# import tensorflow as tf
from sklearn.metrics import classification_report
from nncodec.framework.applications.utils.metrics import get_topk_accuracy_per_batch
from nncodec.framework.applications.models.tokenizer import Tokenizer
from nncodec.nnc_core import nnr_model
from contextlib import nullcontext
import re
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_classification_model(model, criterion=None, testloader=None, testset=None,  min_sample_size=1000, max_batches=None,
                                  early_stopping_threshold=None, device=DEVICE, print_classification_report=False,
                                  return_predictions=False, verbose=False, rec_mdl=False):
    """
    Helper function to evaluate model on test dataset.

    Parameters
    ----------
    model: torch.nn.Module
        Neural network model.
    criterion: torch.nn.Criterion
        Criterion for loss calculation.
    testloader: torch.utils.data.DataLoader
        DataLoader that loaded testset.
    testset: torch.utils.data.dataset.Dataset
        Test dataset
    min_sample_size: int
        Minimum sample size used for early_stopping calculation. Default: 1000
    max_batches: int
        Maximum batches evaluated, by default evaluates the complete testset. Default: None
    early_stopping_threshold: int
        A value between 0-100 corresponding to the accuracy. If it drops under a given threshold
        the evaluation is stopped.
    device: str
        Device on which the model is evaluated: cpu or cuda.
    print_classification_report: bool
        If True print the complete confusion matrix for all the classes.
    return_predictions: bool
        If True return all the predictions for all samples, otherwise return the accuracy.
    verbose: bool
        If True print the progress bar of the evaluation.

    Return
    ------
    output: float | nd.array
        Accuracy or all predictions, depending on the given return_predictions parameter.
    """
    if isinstance(model, nnr_model.ModelExecute):
        criterion = model.handle.criterion
        testloader = model.test_loader
        testset = model.test_set
        device = model.device
        if rec_mdl:
            model = model.rec_model
        else:
            model = model.model

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    model.eval()
    test_loss = []
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    top5_acc = 0

    # set (verbose) iterator
    total_iterations = max_batches or len(testloader)
    # iterator = tqdm(enumerate(testloader), total=total_iterations, position=0, leave=True) if verbose else enumerate(testloader)
    iterator = enumerate(testloader)

    DeepLab_condition = model.__class__.__name__ == "DeepLabV3" if not (isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel)) \
                            else model.module.__class__.__name__ == "DeepLabV3"

    if DeepLab_condition:
        from torchmetrics import JaccardIndex
        num_of_classes = model.classifier[len(model.classifier) - 1].weight.shape[0]
        jaccard = JaccardIndex(task="multiclass", num_classes=num_of_classes, average="macro").to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in iterator:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if DeepLab_condition:
                outputs = outputs['out']
                targets = targets * (targets != 1) * 255
                targets = targets.squeeze(1).long()

            loss = criterion(outputs, targets)

            if outputs.size(1) > 5 and not DeepLab_condition:
                c1, c5 = get_topk_accuracy_per_batch(outputs, targets, topk=(1, 5))
                top5_acc += c5 * targets.size(0)

            test_loss.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0) if not DeepLab_condition else targets.numel()
            correct += predicted.eq(targets).sum().item()
            all_predictions.append(np.array(predicted.cpu()))
            all_labels.append(np.array(targets.cpu()))

            acc = 100. * correct / total

            if DeepLab_condition:
                jaccard.update(predicted, targets)
                if verbose:
                    print('Running Test/Val mIOU (batch {}/{}) over all {} classes: {}'.format(batch_idx,
                                                                                               total_iterations,
                                                                                               num_of_classes,
                                                                                               jaccard.compute() * 100))

            if batch_idx == max_batches:
                break
            elif len(all_predictions) > min_sample_size and early_stopping_threshold is not None and \
                    acc < early_stopping_threshold:
                break

        acc = 100. * correct / total
        if top5_acc != 0:
            top5_acc = top5_acc / total

        if print_classification_report:
            print(classification_report(np.concatenate(all_labels), np.concatenate(all_predictions),
                                        target_names=list(testset.mapping.keys()),
                                        labels=list(testset.mapping.values())))

        if return_predictions:
            return np.concatenate(all_predictions)
        else:
            mean_test_loss = np.mean(test_loss)
            if DeepLab_condition:
                m_IoU = jaccard.compute() * 100
                return {'acc': acc, 'm_IoU': m_IoU, 'mean_test_loss': mean_test_loss}
            else:
                return {'acc': acc, 'top5_acc': float(top5_acc), 'mean_test_loss': mean_test_loss}

def evaluate_classification_model_TEF(model, test_loader, test_set, num_workers=8, verbose=0):

    _ , val_labels = zip(*test_set.imgs)

    y_pred = model.predict(test_loader, verbose=verbose, callbacks=None, max_queue_size=10, workers=num_workers,
                           use_multiprocessing=True)

    top5 = tf.keras.metrics.sparse_top_k_categorical_accuracy(val_labels, y_pred, k=5)
    top1 = tf.keras.metrics.sparse_categorical_accuracy(val_labels, y_pred)
    loss = tf.keras.metrics.sparse_categorical_crossentropy(val_labels, y_pred)

    acc = []
    acc.append((tf.keras.backend.sum(top1) / len(top1)).numpy() * 100)
    acc.append((tf.keras.backend.sum(top5) / len(top5)).numpy() * 100)
    acc.append((tf.keras.backend.mean(loss)).numpy())

    return acc

@torch.no_grad()
def evaluate_language_model(model, testloader, device='mps', max_batches=3, verbose=False, criterion=None, detokenize=False, args=None):

    if detokenize:
        import textwrap
        import json

        def is_number(s):
            try:
                float(s)  # float() can handle both integers and floats
                return True
            except ValueError:
                return False

        enc = Tokenizer(tokenizer_model=f"{args.tokenizer_path}")
        vocab_size = enc.sp_model.get_piece_size()
        vocabulary = [enc.sp_model.id_to_piece(i) for i in range(vocab_size)]

        global_predictions, global_gt, global_mse, global_rel_diff, glob_t_per_sample = {}, {}, {}, {}, {}

    model = model.to(device)

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}["float16"]

    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    )

    if max_batches == None:
        try:
            max_batches = len(testloader)
        except:
            max_batches = testloader.dataset.num_samples

    model.eval()
    test_loss = []
    test_acc = []

    for k, batch in enumerate(testloader):

        if k == max_batches:
            break

        if detokenize:
            X, Y = batch, None
        else:
            X, Y = batch

        X = X.to(device, non_blocking=True)
        if Y is not None:
            Y = Y.to(device, non_blocking=True)
        with ctx:
            logits = model(X,Y)
            loss = model.last_loss

        if not detokenize:
            test_loss.append(loss.item())
            predictions = torch.argmax(logits, dim=-1)  # Shape: (batch_size, seq_length)
            correct = (predictions == Y).float()
            test_acc.append(correct.mean().item() * 100)

        if detokenize:
            output_sample = None
            text_width = 150

            exclude_strings = {'=', '-->', ''}

            def merge_consecutive_digits(token_list):  ## merging consecutiive digits to one number
                merged_digits = []
                i = 0
                while i < len(token_list):
                    if token_list[i] in {'', '='}:
                        merged_digits.append(token_list[i])
                        i += 1
                    elif re.match(r'^-?\d*\.?\d*$', token_list[i]) and token_list[i] != '':
                        num = token_list[i]
                        i += 1
                        while i < len(token_list) and re.match(r'^\d*\.?\d*$', token_list[i]):# and token_list[i] != '':
                            num += token_list[i]
                            i += 1
                        merged_digits.append(num)
                    else:
                        merged_digits.append(token_list[i])
                        i += 1
                return merged_digits

            inp_token = X[0,:]

            wrapped_text = [enc.decode(inp_token[i].int().tolist()) for i in range(X.shape[1])]

            feature_under_test = args.spec_feat_test if args.spec_feat_test is not None else "-->"
            arrow_indices = [i for i, s in enumerate(wrapped_text) if s == feature_under_test]
            if arrow_indices:
                input_sample = wrapped_text[:arrow_indices[-1] + 1]
                input_sample = merge_consecutive_digits(input_sample)
                print(f"Input:\n{textwrap.fill(' '.join(input_sample), width=text_width)} \n")

                output_sample = wrapped_text[arrow_indices[-1]:]
                model_input = inp_token[:arrow_indices[-1] + 1].unsqueeze(0)
                concat_results = [] if feature_under_test == "-->" else [feature_under_test]
                current_line_length = 0
                last_char = ""
                print(f"Predicting [...]\n")
                t_per_token = []
                for i in range(max(args.max_seq_len, len(output_sample) + 1)):
                    if model_input.shape[1] > args.max_seq_len:
                        break

                    t0 = time.perf_counter()
                    logits = model(model_input)

                    tpt = time.perf_counter() - t0
                    t_per_token.append(tpt)
                    
                    arg_max = torch.argmax(logits.squeeze(0), dim=1).item()
                    char = vocabulary[arg_max]

                    if char == 'time':
                        break
                    elif char == "<unk>":
                        break
                    elif char != '▁':
                        concat_results.append(char)
                        print(f'{char if ((is_number(last_char) or last_char in [".", "-"]) and (is_number(char) or char == ".")) else f" {char}"}', end='', flush=True)
                        current_line_length += len(char)
                        last_char = char
                    else:
                        if last_char and not (is_number(last_char) or last_char in [".", "-"]):
                            print(" ", end='', flush=True)
                        current_line_length += 1

                    model_input = torch.cat((model_input, torch.tensor([[arg_max]], device=device)), dim=1)

                    if current_line_length >= text_width:
                        print()
                        current_line_length = 0

            if output_sample:
                print("\n")
                print(f"Ground Truth:\n")
                output_sample = merge_consecutive_digits(output_sample)
                concat_results = merge_consecutive_digits(concat_results)
                print(f"{textwrap.fill(' '.join(output_sample), width=text_width)} \n")

                def extract_values(string_list):
                    values, i = {}, 0
                    while i < len(string_list) - 1:
                        key = string_list[i].strip()
                        if key not in exclude_strings:
                            j = i + 1
                            while j < len(string_list) and string_list[j].strip() in exclude_strings:
                                j += 1
                            if j < len(string_list):
                                try:
                                    value = float(string_list[j].strip())
                                    values[key] = value
                                except ValueError:
                                    pass
                        i += 1
                    return values

                predicted_values = extract_values(concat_results)
                ground_truth_values = extract_values(output_sample)

                rel_differences, diff = {}, {}
                for key in predicted_values:
                    if key in ground_truth_values:
                        diff[key] = predicted_values[key] - ground_truth_values[key]
                        if key in global_predictions:
                            global_predictions[key] += [predicted_values[key]]
                        else:
                            global_predictions[key] = [predicted_values[key]]
                        if key in global_gt:
                            global_gt[key] += [ground_truth_values[key]]
                        else:
                            global_gt[key] = [ground_truth_values[key]]

                for key in diff:
                    rel_differences[key] = (diff[key] / (ground_truth_values[key] + 1e-10) ) * 100
                    if key in global_rel_diff:
                        global_rel_diff[key] += [rel_differences[key]]
                    else:
                        global_rel_diff[key] = [rel_differences[key]]

                print("---------------------------------------------------------------------------------------\n")
                print(f"Relative differences wrt. ground truth: ")
                [print(f"{rel_diff}: {rel_differences[rel_diff]:.1f}%") for rel_diff in rel_differences]
                print("---------------------------------------------------------------------------------------\n")
                print(f"Running absolute mean relative differences wrt. ground truth (global):")
                [print(f"{grel_diff}: {np.mean(np.abs(np.array(global_rel_diff[grel_diff]))):.1f}%") for grel_diff in global_rel_diff]
                print("---------------------------------------------------------------------------------------\n")
                print(f"Time to complete test sequence: {np.sum(t_per_token):.2f}s (avg. time per token:{np.mean(t_per_token):.2f}s)")

                if "tSeq" in glob_t_per_sample:
                    glob_t_per_sample["tSeq"].append(np.sum(t_per_token))
                    glob_t_per_sample["avgTTok"].append(np.mean(t_per_token))
                    glob_t_per_sample["numPredTok"].append(len(t_per_token))
                else:
                    glob_t_per_sample["tSeq"] = [np.sum(t_per_token)]
                    glob_t_per_sample["avgTTok"] = [np.mean(t_per_token)]
                    glob_t_per_sample["numPredTok"] = [len(t_per_token)]

                with open(f'{args.results}/test_predictions.json', 'w') as f:
                    json.dump(global_predictions, f)
                with open(f'{args.results}/test_ground_truth.json', 'w') as f:
                    json.dump(global_gt, f)
                with open(f'{args.results}/test_rel_differences.json', 'w') as f:
                    json.dump(global_rel_diff, f)
                with open(f'{args.results}/test_times.json', 'w') as f:
                    json.dump(glob_t_per_sample, f)

    loss = torch.mean(torch.tensor(test_loss)).item()
    acc = torch.mean(torch.tensor(test_acc)).item()
    ppl = torch.exp(torch.tensor(loss)).item()
    model.train()
    print(f'top1-acc: {acc:.3f}%, ppl: {ppl:.3f}, loss: {loss:.3f}')
    return {'top1_acc': acc, 'ppl': ppl, 'loss': loss}

@torch.no_grad()
def evaluate_mtt(model, testloader, device='mps', max_batches=3, verbose=False, criterion=None, detokenize=False, args=None):

    if detokenize:
        import textwrap
        import json

        # def is_number(s):
        #     try:
        #         float(s)  # float() can handle both integers and floats
        #         return True
        #     except ValueError:
        #         return False

        enc = Tokenizer(tokenizer_model=f"{args.tokenizer_path}")
        vocab_size = enc.sp_model.get_piece_size()
        vocabulary = [enc.sp_model.id_to_piece(i) for i in range(vocab_size)]

        with open(f'{args.tokenizer_path.split("/")[-2]}/mean_dict.json', 'r') as json_file:
            mean_dict = json.load(json_file)
        with open(f'{args.tokenizer_path.split("/")[-2]}/std_dict.json', 'r') as json_file:
            std_dict = json.load(json_file)

        global_predictions, global_gt, global_mse, global_rel_diff, glob_t_per_sample = {}, {}, {}, {}, {}

    model = model.to(device)
    # if detokenize:
    #     class Iterator:
    #         @staticmethod
    #         def iter_batches(dataloader, device):
    #             for x in dataloader:
    #                 x = x.to(device, non_blocking=True)
    #                 yield x
    # else:
    #     class Iterator:
    #         @staticmethod
    #         def iter_batches(dataloader, device):
    #             for x, y in dataloader:
    #                 x = x.to(device, non_blocking=True)
    #                 y = y.to(device, non_blocking=True)
    #                 yield x, y

    exclude_strings = {'=', '-->', ''}

    def denormalize(list_of_str_tokens):
        i = 0
        while i < len(list_of_str_tokens) - 1:
            key = list_of_str_tokens[i].strip()
            if key in mean_dict:
                j = i + 1
                while j < len(list_of_str_tokens) and list_of_str_tokens[j].strip() in exclude_strings:
                    j += 1
                if j < len(list_of_str_tokens):
                    try:
                        original_value = float(list_of_str_tokens[j].strip())
                        updated_value = (original_value * std_dict[key]) + mean_dict[key]
                        list_of_str_tokens[j] = f"{updated_value:.3f}"
                    except ValueError:
                        pass
            i += 1

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}["float16"]

    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    )

    # test_batch_iter = []
    # batch_iter = Iterator.iter_batches(testloader, device)

    if max_batches == None:
        try:
            max_batches = len(testloader)
        except:
            max_batches = testloader.dataset.num_samples

    # for idx, x in enumerate(batch_iter):
    #     if idx >= max_batches:
    #         break
    #     test_batch_iter.append(x)

    model.eval()
    test_loss = []
    test_acc = []
    test_mse = []

    losses_class = torch.zeros(max_batches)
    losses_reg = torch.zeros(max_batches)
    losses_dec = torch.zeros(max_batches)

    for k, batch in enumerate(testloader):

        if k == max_batches:
            break

        if detokenize:
            X, Y = batch, None
        else:
            X, Y = batch

        X = X.to(device, non_blocking=True)
        if Y is not None:
            Y = Y.to(device, non_blocking=True)
        with ctx:
            logits, reg_out, gating_out, backbone_output = model(X, Y)
            loss = model.class_loss
            loss_reg = model.loss_reg
            loss_dec = model.decision_loss

        if not detokenize:
            losses_class[k] = loss.item()
            losses_reg[k] = loss_reg.item()
            losses_dec[k] = loss_dec.item()
            test_loss.append(loss.item())

            cond_mask = Y[:, :, 0]
            mask_class = (cond_mask == 0)
            mask_reg = (cond_mask == 1)

            gt_words = Y[:, :, 1][mask_class]
            gt_num = Y[:, :, 1][mask_reg]
            word_predictions = torch.argmax(logits, dim=-1)[mask_class]  # Shape: (batch_size, seq_length)
            correct = (word_predictions == gt_words).float()
            test_acc.append(correct.mean().item() * 100)

            torch.set_printoptions(sci_mode=False)
            diff = (gt_num - reg_out[:,:,0][mask_reg])
            mse = torch.mean(diff ** 2).item()
            test_mse.append(mse)

        if detokenize:
            text_width = 150

            exclude_strings = {'=', '-->', ''}

            def merge_consecutive_digits(token_list):  ## merging consecutiive digits to one number
                merged_digits = []
                i = 0
                while i < len(token_list):
                    if token_list[i] in {'', '='}:
                        merged_digits.append(token_list[i])
                        i += 1
                    elif re.match(r'^-?\d*\.?\d*$', token_list[i]) and token_list[i] != '':
                        num = token_list[i]
                        i += 1
                        while i < len(token_list) and re.match(r'^\d*\.?\d*$', token_list[i]) and token_list[i] != '':
                            num += token_list[i]
                            i += 1
                        merged_digits.append(num)
                    else:
                        merged_digits.append(token_list[i])
                        i += 1
                return merged_digits

            # for batch_elemet in range(X.shape[0]):
            inp_token = X[0,:,:]
            numbers = (inp_token[inp_token[:, 0] == 1][:, 1])
            number_iterator = iter(numbers)

            wrapped_text = [" " + str(next(number_iterator).item()) + " " if inp_token[i, 0] == 1 else enc.decode(
                            inp_token[i, 1].int().tolist()) for i in range(inp_token.shape[0])]

            # print(f"Input:\n{textwrap.fill(' '.join(wrapped_text), width=text_width)} \n")

            feature_under_test = args.spec_feat_test if args.spec_feat_test is not None else "-->"
            arrow_indices = [i for i, s in enumerate(wrapped_text) if s == feature_under_test]

            input_sample = wrapped_text[:arrow_indices[-1] + 1]
            if numbers.numel() != 0:
                denormalize(input_sample)
            else:
                input_sample = merge_consecutive_digits(input_sample)
            print(f"Input:\n{textwrap.fill(' '.join(input_sample), width=text_width)} \n")

            output_sample = wrapped_text[arrow_indices[-1]:]
            model_input = inp_token[:arrow_indices[-1] + 1,:].unsqueeze(0)
            concat_results = [] if feature_under_test == "-->" else [feature_under_test]
            current_line_length = 0
            print(f"Predicting [...]\n")

            t_per_token = []
            tokens_to_predict = max(args.max_seq_len, len(output_sample) + 1) if (args.spec_feat_test is None) else 4
            for i in range(tokens_to_predict):
                if model_input.shape[1] > args.max_seq_len:
                    break

                t0 = time.perf_counter()

                logits, reg_out, gating_out, backbone_output = model(model_input)

                tpt = time.perf_counter() - t0
                t_per_token.append(tpt)

                # number_condition = (len(concat_results) > 2 and concat_results[-1] == "=" and concat_results[-2] in mean_dict)
                # fill_in_the_gap_condition = number_condition and len(output_sample) > output_sample.index(concat_results[-2]) + 4 and not(is_number(output_sample[output_sample.index(concat_results[-2]) + 4]))
                if numbers.numel() == 0 or not gating_out:#not number_condition and not fill_in_the_gap_condition:
                    arg_max = torch.argmax(logits.squeeze(0), dim=1).item()
                    char = vocabulary[arg_max]

                    if char == 'time':
                        break
                    elif char == "<unk>":
                        break
                    elif char != '▁':
                        concat_results.append(char)
                        print(char, end='', flush=True)
                        current_line_length += len(char)
                    else:
                        print(" ", end='', flush=True)
                        current_line_length += 1

                    model_input = torch.cat((model_input, torch.tensor([[[0.0, arg_max]]], device=device)), dim=1)

                else:
                    filtered_res = [s for s in concat_results if s not in exclude_strings]
                    if filtered_res[-1] in std_dict and filtered_res[-1] in mean_dict:
                        reg_out_denorm = (reg_out * std_dict[filtered_res[-1]]) + mean_dict[filtered_res[-1]]

                    reg_out_str = f"{reg_out_denorm.item():.3f}" #if not fill_in_the_gap_condition else f"[{reg_out_denorm.item():.3f}]"

                    print(reg_out_str, end='', flush=True)
                    concat_results.append(reg_out_str)
                    current_line_length += len(reg_out_str)

                    model_input = torch.cat((model_input, torch.tensor([[[1.0, reg_out[0, 0, 0]]]], device=device)), dim=1)

                if current_line_length >= text_width:
                    print()
                    current_line_length = 0

            print("\n")
            print(f"Ground Truth:\n")
            if numbers.numel() != 0:
                denormalize(output_sample)
            else:

                output_sample = merge_consecutive_digits(output_sample)
                concat_results = merge_consecutive_digits(concat_results)
            print(f"{textwrap.fill(' '.join(output_sample), width=text_width)} \n")

            def extract_values(string_list):
                values, i = {}, 0
                while i < len(string_list) - 1:
                    key = string_list[i].strip()
                    if key not in exclude_strings:
                        j = i + 1
                        while j < len(string_list) and string_list[j].strip() in exclude_strings:
                            j += 1
                        if j < len(string_list):
                            try:
                                value = float(string_list[j].strip())
                                values[key] = value
                            except ValueError:
                                pass
                    i += 1
                return values

            predicted_values = extract_values(concat_results)
            ground_truth_values = extract_values(output_sample)

            rel_differences, diff = {}, {}
            for key in predicted_values:
                if key in ground_truth_values:
                    diff[key] = predicted_values[key] - ground_truth_values[key]
                    if key in global_predictions:
                        global_predictions[key] += [predicted_values[key]]
                    else:
                        global_predictions[key] = [predicted_values[key]]
                    if key in global_gt:
                        global_gt[key] += [ground_truth_values[key]]
                    else:
                        global_gt[key] = [ground_truth_values[key]]

            for key in diff:
                rel_differences[key] = (diff[key] / (ground_truth_values[key] + 1e-10) ) * 100
                if key in global_rel_diff:
                    global_rel_diff[key] += [rel_differences[key]]
                else:
                    global_rel_diff[key] = [rel_differences[key]]

            print("---------------------------------------------------------------------------------------\n")
            print(f"Relative differences wrt. ground truth: ")
            [print(f"{rel_diff}: {rel_differences[rel_diff]:.1f}%") for rel_diff in rel_differences]
            print("---------------------------------------------------------------------------------------\n")
            print(f"Running absolute mean relative differences wrt. ground truth (global):")
            [print(f"{grel_diff}: {np.mean(np.abs(np.array(global_rel_diff[grel_diff]))):.1f}%") for grel_diff in global_rel_diff]
            print("---------------------------------------------------------------------------------------\n")
            print(f"Time to complete test sequence: {np.sum(t_per_token):.2f}s (avg. time per token: {np.mean(t_per_token):.2f}s)")

            if "tSeq" in glob_t_per_sample:
                glob_t_per_sample["tSeq"].append(np.sum(t_per_token))
                glob_t_per_sample["avgTTok"].append(np.mean(t_per_token))
                glob_t_per_sample["numPredTok"].append(len(t_per_token))
            else:
                glob_t_per_sample["tSeq"] = [np.sum(t_per_token)]
                glob_t_per_sample["avgTTok"] = [np.mean(t_per_token)]
                glob_t_per_sample["numPredTok"] = [len(t_per_token)]


            with open(f'{args.results}/test_predictions.json', 'w') as f:
                json.dump(global_predictions, f)
            with open(f'{args.results}/test_ground_truth.json', 'w') as f:
                json.dump(global_gt, f)
            with open(f'{args.results}/test_rel_differences.json', 'w') as f:
                json.dump(global_rel_diff, f)
            with open(f'{args.results}/test_times.json', 'w') as f:
                json.dump(glob_t_per_sample, f)


    out_class = losses_class.mean()
    out_reg = losses_reg.mean()
    out_dec = losses_dec.mean()

    loss = torch.mean(torch.tensor(test_loss)).item()
    acc = torch.mean(torch.tensor(test_acc)).item()
    ppl = torch.exp(out_class).item()
    mse = torch.mean(torch.tensor(test_mse)).item()
    model.train()
    # return acc, ppl, loss.item()
    print(f'top1-acc (words): {acc:.3f}%, ppl: {ppl:.3f}, mse (numbers): {mse:.3f}')
    return {'test_out_class': out_class, 'test_out_reg': out_reg, 'test_out_dec': out_dec}
