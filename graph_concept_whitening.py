import os
import torch
import pickle
from plot_functions_graphs import (concept_gradient_importance, plot_concept_gradient_importance,
                                   intra_concept_dot_product_vs_inter_concept_dot_product, plot_elegant_concept_similarities,
                                   compute_axis_alignment_accuracy_and_f1, extract_top_activation_subgraphs_per_concept,
                                   draw_graph)

from graph_classification import (save_checkpoint, generate_classification_report, evaluate_model_performance,
                                  get_model_and_device, get_loaders, get_optimizer_and_criterion)


def train_concept_whitening(model, loader, concept_loaders, optimizer, criterion, device):
    model = model.to(device)
    model.train()
    neg_con_align = 0.0

    for graph_batch_index, graph_batch in enumerate(loader):
        if (graph_batch_index + 1) % 3 == 0:
            model.eval()
            with torch.no_grad():
                # update the gradient matrix G
                for concept_index, (concept, concept_loader) in enumerate(concept_loaders.items()):
                    model.change_mode(concept_index)
                    for concept_batch_index, concept_batch in enumerate(concept_loader):
                        concept_batch.x = torch.autograd.Variable(concept_batch.x)
                        model(concept_batch.x.to(device), concept_batch.edge_index.to(device), concept_batch.batch.to(device))
                        break
                neg_con_align = model.update_rotation_matrix()
                print(f"negative concept alignment:{neg_con_align}")
                # change to ordinary mode
                model.change_mode(-1)
            model.train()

        out = model(graph_batch.x.to(device), graph_batch.edge_index.to(device), graph_batch.batch.to(device))
        loss = criterion(out, graph_batch.y.to(device))
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
        optimizer.step()
    return neg_con_align


def concept_whitening_epoch_iterator(dataset, classes, graphs_dataset_prefix, graph_concepts_dataset_prefix,
                                     concepts, graph_model_path, graph_conv_type, graph_residual_connections,
                                     whitened_graph_model_path, concept_type):

    train_loader, test_loader, train_concept_loaders, test_concept_loaders = get_loaders(dataset, classes,
                                                                                         graphs_dataset_prefix,
                                                                                         graph_concepts_dataset_prefix + '_' + concept_type, concepts)

    model, device, last_epoch, best_test_acc = get_model_and_device(train_loader.dataset, len(classes), graph_model_path,
                                                                    graph_conv_type, graph_residual_connections,
                                                                    whitening=True)

    optimizer, criterion = get_optimizer_and_criterion(model, graph_model_path)

    if last_epoch != 0:
        train_acc = evaluate_model_performance(model, train_loader, device)
        test_acc = evaluate_model_performance(model, test_loader, device)
        print(f"Resuming training from Epoch: {last_epoch:03d}, Best Test Acc: {best_test_acc:.4f}")

    early_stop_counter = 0
    best_neg_con_align = 1000

    for epoch in range(last_epoch + 1, last_epoch + 1000):
        neg_con_align = train_concept_whitening(model, train_loader, train_concept_loaders, optimizer, criterion, device)
        train_acc = evaluate_model_performance(model, train_loader, device)
        test_acc = evaluate_model_performance(model, test_loader, device)
        # train_acc_report = generate_classification_report(classes, model, train_loader, device)
        # test_acc_report = generate_classification_report(classes, model, test_loader, device)
        # print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Acc Report: {train_acc_report:.4f}, Test Acc: {test_acc:.4f}, Test Acc Report: {test_acc_report:.4f}")

        print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        if neg_con_align < best_neg_con_align:
            early_stop_counter = 0
            best_neg_con_align = neg_con_align

            print(f"New best Negative Concept Alignment achieved on Epoch: {epoch:03d}, "
                  f"Negative Concept Alignment: {best_neg_con_align}, "
                  f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}.")

            print(f"Saving checkpoint.")

            save_checkpoint({'epoch': epoch, 'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'best_test_acc': test_acc,
                             'best_neg_con_align': best_neg_con_align}, whitened_graph_model_path)

        else:
            early_stop_counter += 1

        if early_stop_counter == 5:
            print("Concept whitening training is stagnating, so training is stopped.")
            break


def concept_whitening_inference(dataset, classes, graphs_dataset_prefix, graph_concepts_dataset_prefix, concepts,
                                whitened_graph_model_path, graph_conv_type, graph_residual_connections, concept_type):

    train_loader, test_loader, train_concept_loaders, test_concept_loaders = get_loaders(dataset, classes,
                                                                                         graphs_dataset_prefix,
                                                                                         graph_concepts_dataset_prefix + '_' + concept_type,
                                                                                         concepts)

    model, device, last_epoch, best_test_acc, best_neg_con_align = get_model_and_device(train_loader.dataset, len(classes),
                                                                                        whitened_graph_model_path,
                                                                                        graph_conv_type, graph_residual_connections,
                                                                                        whitening=True)

    print("Inference classification report")
    generate_classification_report(classes, model, test_loader, device)
    print(f"Final test accuracy inference: {evaluate_model_performance(model, test_loader, device)}")
    print(f"Final negative concept alignment: {best_neg_con_align}")


def aggregate_concept_gradient_importance_data(dataset, classes, graphs_dataset_prefix, concepts, graph_conv_type,
                                               graph_residual_connections, whitened_graph_model_path, target_class):

    train_loader, test_loader = get_loaders(dataset, classes, graphs_dataset_prefix,
                                            concept_gradient_importance_flag=True)

    model, device, last_epoch, best_test_acc, best_neg_con_align = get_model_and_device(train_loader.dataset, len(classes), whitened_graph_model_path,
                                                                    graph_conv_type, graph_residual_connections,
                                                                    whitening=True)

    concepts_importances = concept_gradient_importance(model, test_loader, classes, concepts,
                                                       class_index=test_loader.dataset.class_to_idx(target_class),
                                                       device=device)

    return concepts_importances


def aggregate_doc_cap_data(dataset, classes, graphs_dataset_prefix, concepts, graph_conv_type,
                           graph_residual_connections, whitened_graph_model_path):

    train_loader, test_loader = get_loaders(dataset, classes, graphs_dataset_prefix)

    model, device, last_epoch, best_test_acc, best_neg_con_align = get_model_and_device(
        train_loader.dataset, len(classes), whitened_graph_model_path,
        graph_conv_type, graph_residual_connections, whitening=True)

    axis_alignment = {}

    for axis_idx, concept_name in enumerate(concepts):
        orig_label = classes.index(concept_name)

        f1 = compute_axis_alignment_accuracy_and_f1(model, device, test_loader,
                                                    orig_label=orig_label, axis_index=axis_idx, whitening=True)

        axis_alignment[concept_name] = f1

    print(axis_alignment)

    for name, f1 in axis_alignment.items():
        print(f"{name:15s} -  F1: {f1:.3f}")

    f1s = list(axis_alignment.values())
    mean_f1 = sum(f1s) / len(f1s)

    print(f"Overall doc-CAP mean F1: {mean_f1}")


def aggregate_concept_dot_product_data(dataset, classes, graphs_dataset_prefix, graph_concepts_dataset_prefix,
                                       concepts, graph_model_path, graph_conv_type, graph_residual_connections,
                                       whitened_graph_model_path, concept_type, whitening=False):

    train_loader, merged_test_concept_loader = get_loaders(dataset, classes, graphs_dataset_prefix,
                                                           graph_concepts_dataset_prefix + '_' + concept_type, concepts,
                                                           concept_dot_product_flag=True)

    if whitening:
        model, device, last_epoch, best_test_acc, best_neg_con_align = get_model_and_device(train_loader.dataset,
                                                                                            len(classes),
                                                                                            whitened_graph_model_path,
                                                                                            graph_conv_type,
                                                                                            graph_residual_connections,
                                                                                            whitening=True)

    else:
        model, device, last_epoch, best_test_acc = get_model_and_device(train_loader.dataset, len(classes),
                                                                        graph_model_path, graph_conv_type,
                                                                        graph_residual_connections)

    dot_product_matrix = intra_concept_dot_product_vs_inter_concept_dot_product(model, merged_test_concept_loader,
                                                                                concepts, classes, device,
                                                                                whitening=whitening)

    return dot_product_matrix


def aggregate_concept_axis_visualization_data(dataset, classes, graphs_dataset_prefix, graph_concepts_dataset_prefix,
                                              concepts, concept_type, graph_conv_type, graph_residual_connections,
                                              whitened_graph_model_path, graph_model_path, whitening=False):

    train_loader, merged_test_concept_loader = get_loaders(dataset, classes, graphs_dataset_prefix,
                                                           graph_concepts_dataset_prefix + '_' + concept_type, concepts,
                                                           concept_concept_axis_visualization_flag=True)

    if whitening:
        model, device, last_epoch, best_test_acc, best_neg_con_align = get_model_and_device(train_loader.dataset,
                                                                                            len(classes),
                                                                                            whitened_graph_model_path,
                                                                                            graph_conv_type,
                                                                                            graph_residual_connections,
                                                                                            whitening=True)

    else:
        model, device, last_epoch, best_test_acc = get_model_and_device(train_loader.dataset, len(classes),
                                                                        graph_model_path, graph_conv_type,
                                                                        graph_residual_connections)

    axis_alignment = {}

    for axis_idx, concept_name in enumerate(concepts):
        # find the original numeric label for this concept
        orig_label = classes.index(concept_name)

        f1 = compute_axis_alignment_accuracy_and_f1(model, device, merged_test_concept_loader,
                                                         orig_label=orig_label, axis_index=axis_idx, whitening=whitening)

        axis_alignment[concept_name] = f1

    print(axis_alignment)

    # print per‐class results
    for name, f1 in axis_alignment.items():
        print(f"{name:15s} →  F1: {f1:.3f}")

    # now compute the overall means
    f1s = [metric for metric in axis_alignment.values()]

    mean_f1 = sum(f1s) / len(f1s)

    print(f"Overall axis‐alignment mean F1: {mean_f1}")


def aggregate_top_activation_subgraphs_per_concept_data(dataset, classes, images_prefix, graphs_dataset_prefix,
                                                        graph_concepts_dataset_prefix, concepts, concept_type,
                                                        graph_conv_type, graph_residual_connections,
                                                        whitened_graph_model_path):

    train_loader, test_loader, train_concept_loaders, test_concept_loaders = get_loaders(dataset, classes,
                                                                                         graphs_dataset_prefix,
                                                                                         graph_concepts_dataset_prefix +
                                                                                         '_' + concept_type, concepts)

    model, device, last_epoch, best_test_acc, best_neg_con_align = get_model_and_device(train_loader.dataset,
                                                                                        len(classes),
                                                                                        whitened_graph_model_path,
                                                                                        graph_conv_type,
                                                                                        graph_residual_connections,
                                                                                        whitening=True)

    for axis_idx, concept_name in enumerate(concepts):
        loader = test_concept_loaders[concept_name]
        best_file_path, best_val = extract_top_activation_subgraphs_per_concept(model, device, loader, axis_index=axis_idx)
        print(f"Concept `{concept_name}`  →  top subgraph: `{os.path.basename((best_file_path))}`  file path: `{best_file_path}` (activation = {best_val:.4f})")
        draw_graph(images_prefix, subgraph_path=best_file_path)


def whiten_graph_concepts(dataset, classes, images_prefix, graphs_dataset_prefix, graph_concepts_dataset_prefix, concepts,
                          graph_model_path, graph_conv_type, graph_residual_connections, whitened_graph_model_paths,
                          concept_type, negative_concept_types, target_class, mode):

    if mode == 'train':

        print(f"Training graph concept whitening for {dataset} using {concept_type} for GNN {graph_conv_type} "
              f"with residual connections {graph_residual_connections}")

        concept_whitening_epoch_iterator(dataset, classes, graphs_dataset_prefix, graph_concepts_dataset_prefix,
                                         concepts, graph_model_path, graph_conv_type, graph_residual_connections,
                                         whitened_graph_model_paths[concept_type], concept_type)

    elif mode == 'predict':

        print(f"Predicting graph classification for {dataset} using GNN {graph_conv_type} residual connections "
              f"{graph_residual_connections} with Concept Whitening for {concept_type} graph concepts.")

        concept_whitening_inference(dataset, classes, graphs_dataset_prefix, graph_concepts_dataset_prefix, concepts,
                                    whitened_graph_model_paths[concept_type], graph_conv_type, graph_residual_connections,
                                    concept_type)

    elif mode == 'concept_gradient_importance':

        aggregate_concept_importances = {}

        aggregate_concept_importances[concept_type] = aggregate_concept_gradient_importance_data(dataset, classes, graphs_dataset_prefix,
                                                                                                 concepts, graph_conv_type, graph_residual_connections,
                                                                                                 whitened_graph_model_paths[concept_type], target_class)

        for negative_concept_type in negative_concept_types:

            aggregate_concept_importances[negative_concept_type] = aggregate_concept_gradient_importance_data(dataset, classes, graphs_dataset_prefix,
                                                                                                              concepts, graph_conv_type, graph_residual_connections,
                                                                                                              whitened_graph_model_paths[negative_concept_type],
                                                                                                              target_class)

        plot_concept_gradient_importance(aggregate_concept_importances, concepts, target_class, dataset, graph_conv_type, graph_residual_connections, images_prefix)

    elif mode == 'concept_dot_product':
        dot_product_matrix_black = aggregate_concept_dot_product_data(dataset, classes, graphs_dataset_prefix,
                                                                      graph_concepts_dataset_prefix,
                                                                      concepts, graph_model_path, graph_conv_type,
                                                                      graph_residual_connections,
                                                                      whitened_graph_model_paths[concept_type],
                                                                      concept_type)

        plot_elegant_concept_similarities(dot_product_matrix_black, concepts, concept_type, 'black-box', dataset,
                                          graph_conv_type, graph_residual_connections, images_prefix)

        dot_product_matrix_white = aggregate_concept_dot_product_data(dataset, classes, graphs_dataset_prefix,
                                                                      graph_concepts_dataset_prefix,
                                                                      concepts, graph_model_path, graph_conv_type,
                                                                      graph_residual_connections,
                                                                      whitened_graph_model_paths[concept_type],
                                                                      concept_type, whitening=True)

        plot_elegant_concept_similarities(dot_product_matrix_white, concepts, concept_type, 'white-box', dataset,
                                          graph_conv_type, graph_residual_connections, images_prefix)

        for negative_concept_type in negative_concept_types:
            dot_product_matrix_black = aggregate_concept_dot_product_data(dataset, classes, graphs_dataset_prefix,
                                                                          graph_concepts_dataset_prefix,
                                                                          concepts, graph_model_path, graph_conv_type,
                                                                          graph_residual_connections,
                                                                          whitened_graph_model_paths[negative_concept_type],
                                                                          negative_concept_type)

            plot_elegant_concept_similarities(dot_product_matrix_black, concepts, negative_concept_type, 'black-box', dataset,
                                              graph_conv_type, graph_residual_connections, images_prefix)

            dot_product_matrix_white = aggregate_concept_dot_product_data(dataset, classes, graphs_dataset_prefix,
                                                                          graph_concepts_dataset_prefix,
                                                                          concepts, graph_model_path, graph_conv_type,
                                                                          graph_residual_connections,
                                                                          whitened_graph_model_paths[negative_concept_type],
                                                                          negative_concept_type, whitening=True)

            plot_elegant_concept_similarities(dot_product_matrix_white, concepts, negative_concept_type, 'white-box', dataset,
                                              graph_conv_type, graph_residual_connections, images_prefix)

    elif mode == 'concept_axis_visualization':
        print(f"Concept alignment metrics for {dataset} using GNN {graph_conv_type} residual connections {graph_residual_connections} and concept type {concept_type} for white-box model.")

        aggregate_concept_axis_visualization_data(dataset, classes, graphs_dataset_prefix,
                                                  graph_concepts_dataset_prefix,
                                                  concepts, concept_type, graph_conv_type, graph_residual_connections,
                                                  whitened_graph_model_paths[concept_type], graph_model_path, whitening=True)

        for negative_concept_type in negative_concept_types:
            print(f"Concept alignment metrics for {dataset} using GNN {graph_conv_type} residual connections {graph_residual_connections} and concept type {negative_concept_type} for white-box model.")

            aggregate_concept_axis_visualization_data(dataset, classes, graphs_dataset_prefix, graph_concepts_dataset_prefix,
                                                      concepts, negative_concept_type, graph_conv_type, graph_residual_connections,
                                                      whitened_graph_model_paths[negative_concept_type], graph_model_path, whitening=True)

        print(f"Concept alignment metrics for {dataset} using GNN {graph_conv_type} residual connections {graph_residual_connections} and concept type {concept_type} for black-box model.")

        aggregate_concept_axis_visualization_data(dataset, classes, graphs_dataset_prefix,
                                                  graph_concepts_dataset_prefix,
                                                  concepts, concept_type, graph_conv_type, graph_residual_connections,
                                                  whitened_graph_model_paths[concept_type], graph_model_path,
                                                  whitening=False)

        for negative_concept_type in negative_concept_types:
            print(f"Concept alignment metrics for {dataset} using GNN {graph_conv_type} residual connections {graph_residual_connections} and concept type {negative_concept_type} for black-box model.")

            aggregate_concept_axis_visualization_data(dataset, classes, graphs_dataset_prefix,
                                                      graph_concepts_dataset_prefix,
                                                      concepts, negative_concept_type, graph_conv_type,
                                                      graph_residual_connections,
                                                      whitened_graph_model_paths[negative_concept_type],
                                                      graph_model_path, whitening=False)

    elif mode == 'doc_cap':
        print(f"Document-level CAP for {dataset} using GNN {graph_conv_type} residual connections {graph_residual_connections} and concept type {concept_type}.")

        aggregate_doc_cap_data(dataset, classes, graphs_dataset_prefix, concepts, graph_conv_type,
                               graph_residual_connections, whitened_graph_model_paths[concept_type])

    elif mode == 'top_activation_subgraphs':
        print(f"Extracting top activating subgraphs for {dataset} using GNN {graph_conv_type} residual connections {graph_residual_connections} and concept type {concept_type} for white-box model.")

        aggregate_top_activation_subgraphs_per_concept_data(dataset, classes, images_prefix, graphs_dataset_prefix,
                                                            graph_concepts_dataset_prefix,
                                                            concepts, concept_type, graph_conv_type,
                                                            graph_residual_connections,
                                                            whitened_graph_model_paths[concept_type])

        for negative_concept_type in negative_concept_types:
            print(f"Extracting top activating subgraphs for {dataset} using GNN {graph_conv_type} residual connections {graph_residual_connections} and concept type {negative_concept_type} for white-box model.")

            aggregate_top_activation_subgraphs_per_concept_data(dataset, classes, images_prefix, graphs_dataset_prefix,
                                                                graph_concepts_dataset_prefix,
                                                                concepts, negative_concept_type, graph_conv_type,
                                                                graph_residual_connections,
                                                                whitened_graph_model_paths[negative_concept_type])
