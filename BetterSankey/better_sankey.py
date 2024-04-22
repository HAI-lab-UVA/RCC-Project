import json
import scipy
from time import time
import numpy as np
import pandas as pd
from collections.abc import Callable

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator


class SankeyPlot: 
    def __init__(self) -> None:
        self.labels = []
        self.sources = []
        self.targets = []
        self.values = []
        self.x_pos = []
        self.y_pos = []
        self.p_values = []
        self.link_colors = []
        self.link_labels = []
        self.hover_colors = []
        self.response_vals = []
        self.val_counts = []
        self.x_reference = []
        self.color_swatch = []
        self.hover_swatch = []
        self.branch_feats = None

    def recurse_sankey_branch(self, branch_mask:pd.Series, start_index:int=0, y_min:float=0.0, y_max:float=1.0,
                              branch_source:int=None, branch_condition:str="") -> None:
        # TODO:
        # * Group links by color (adjust the order added to list to do all of the same response level first)
        # * Display prop. of response levels on hover for feature bar
        for i in range(start_index, len(self.feats)):
            feat = self.feats[i]
            data = self.data[branch_mask]
            self.val_counts.append(data[feat].value_counts(dropna=False).sort_values(ascending=False))
            unique_vals = list(self.val_counts[-1].index)

            # Label w/ feature values plus any branch conditions
            level_labels = [feat + ': ' + str(val) + branch_condition for val in unique_vals]
            self.labels.extend(level_labels)

            # X-positions stay the same regardless of whether we are in a branch
            self.x_pos.extend(np.full(len(unique_vals), self.x_reference[i]))

            # Accurately calculate y-position between 0 and 1
            level_counts = self.val_counts[-1].to_list()
            # Calculate relative proportions of each level
            level_props = [count / sum(level_counts) for count in level_counts]
            # Assign y-position of node center based on the hight (proportion) of each level
            y_align = [1 - (sum(level_props[:n]) + (val / 2)) for n, val in enumerate(level_props)]
            # Reverse list to maintain order
            y_align = y_align[::-1]
            # Account for y_pos in branches w/ ((y_max - y_min) * y_align) + min
            self.y_pos.extend([((y_max - y_min) * y) + y_min for y in y_align])

            if (feat != self.response):
                # Do chi^2 tests for feat vs response
                contingency_tab = pd.crosstab(data[feat], data[self.response], margins=False, dropna=False)
                x2_res = scipy.stats.chi2_contingency(contingency_tab)
                self.p_values.append(x2_res.pvalue)

            if (i > 0):
                # Define links between nodes
                begin_branch = False
                prior_unique_vals = list(self.val_counts[-2].index)
                if (i == start_index and branch_source is not None):
                    # Immediately after branch, there will only be one prior unique val
                    begin_branch = True
                    # We don't care what this value is, we just need a list with one element
                    prior_unique_vals = [None]

                for j, value in enumerate(unique_vals):
                    for k, prior_value in enumerate(prior_unique_vals):
                        for x, response_value in enumerate(self.response_vals):
                            # Count num samples with both values
                            if pd.isnull(value): 
                                value_mask = data[feat].isna()
                            else: value_mask = data[feat] == value
                            
                            if pd.isnull(prior_value): 
                                prior_value_mask = data[self.feats[i-1]].isna()
                            else: prior_value_mask = data[self.feats[i-1]] == prior_value

                            if pd.isnull(response_value): 
                                response_value_mask = data[self.response].isna()
                            else: response_value_mask = data[self.response] == response_value

                            if (begin_branch == True):
                                # Set values: prop of samples that meet the branch conditions, ignoring value of prior node
                                self.values.append(len(self.data[(value_mask) & (response_value_mask) & (branch_mask)]) / len(self.data))
                                # Set source: just the source of this branch
                                self.sources.append(branch_source)
                            else:
                                # Set values: prop of samples in whole dataset, not branch subset
                                self.values.append(len(self.data[(value_mask) & (prior_value_mask) & (response_value_mask) & (branch_mask)]) / len(self.data))
                                # Set source: index labels by -1 * (len(val_counts) + len(prior_val_counts) - k)
                                self.sources.append(self.labels.index(self.labels[-1 * (len(unique_vals) + len(prior_unique_vals) - k)]))

                            # Set target: index of this level's label
                            self.targets.append(self.labels.index(level_labels[j]))
                            self.link_colors.append(self.color_swatch[x])
                            self.link_labels.append(self.response + ": " + str(response_value))
                            self.hover_colors.append(self.hover_swatch[x])

                if (self.branch_feats is not None and feat in self.branch_feats):
                    # Recurse on each level of branch_feat
                    for j, value in enumerate(unique_vals):
                        # Set y_min and y_max relative to height/prop of feat level
                        branch_y_max = y_max - sum(level_props[:j])
                        branch_y_min = y_min + sum(level_props[j+1:])
                        # Get the index for the branch's source
                        new_branch_source = self.labels.index(level_labels[j])
                        # Update branch_mask
                        if pd.isnull(value): 
                            new_branch_mask = branch_mask & self.data[feat].isna()
                        else: new_branch_mask = branch_mask & (self.data[feat] == value)
                        self.recurse_sankey_branch(new_branch_mask, i+1, branch_y_max, branch_y_min, new_branch_source,
                                                   "\n" + level_labels[j])
                    return


    def build_sankey(self, data:pd.DataFrame, feats:list, response:str, 
                     color_swatch:list, hover_swatch:list, branch_feats:list=None,
                     significance=True, color_json:str=None, vertical_pad=15) -> go.Figure:
        
        # TODO: Decide whether to reset these when func called
        # labels = []
        # sources = []
        # targets = []
        # values = []
        # x_pos = []
        # y_pos = []
        # p_values = []
        # link_colors = []
        # link_labels = []
        # hover_colors = []
        # val_counts = []
        self.data = data.reset_index(drop=True)
        self.feats = feats
        self.response = response
        self.response_vals = list(data[response].value_counts(dropna=False).sort_values(ascending=False).index)
        self.x_reference = np.linspace(0.0, 1.0, num=len(feats)).tolist()
        self.color_swatch = color_swatch
        self.hover_swatch = hover_swatch
        self.branch_feats = branch_feats
        dummy_mask = pd.Series([True]).repeat(len(data)).reset_index(drop=True)

        self.recurse_sankey_branch(dummy_mask)
        
        # Create sankey/alluvial diagram
        colors = None
        if (color_json is not None): 
            # Load color dict
            with open(color_json) as fp:
                color_dict = json.load(fp)

            colors = [color_dict[label.split(':')[0]] for label in self.labels[:-len(self.response_vals)]]
            colors.extend(hover_swatch[:len(self.response_vals)])

        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            valueformat = ".2%",
            node = dict(
                pad = vertical_pad,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = self.labels,
                color = colors,
                x = self.x_pos,
                y = self.y_pos,
            ),
            link = dict(
                source = self.sources, # indices correspond to labels
                target = self.targets,
                value = self.values,
                label = self.link_labels,
                color = self.link_colors,
                # BUG: In Plotly package, does not update colors
                hovercolor = self.hover_colors,
            ))])

        if (significance):
            # Add chi^2 p-vals as annotations
            # NOTE: Assumes response is last (x_reference[-1])
            for i, p in enumerate(self.p_values):
                fig.add_annotation(
                    text = "p-value: {:.3e}".format(p),
                    # BUG: index out of range for branching
                    x = self.x_reference[i],
                    # TODO: Offset y for branches
                    y = 0,
                    yshift = -50,
                    showarrow = False,
                )

        fig.update_layout(title_text="Sankey Diagram w/ response: {}".format(response), font_size=10, height=500, width=300 * len(feats))
        return fig

    def run_sfs(self, model:BaseEstimator, train:pd.DataFrame, labels:pd.Series, key:str, results_path:str, 
                direction:str='forward', n_features:int|float='auto', tol:float=None,
                retrain=True) -> BaseEstimator:
        
        with open(results_path) as fp:
            results = json.load(fp)

        tic_fwd = time()
        # NOTE: Always runs all processes in parallel
        sfs = SequentialFeatureSelector(
            model, n_features_to_select=n_features, tol=tol, direction=direction, n_jobs=-1
        ).fit(train, labels)
        toc_fwd = time()
        print(f"Done in {toc_fwd - tic_fwd:.3f}s")

        results[key] = sfs.get_feature_names_out().tolist()
        with open(results_path, 'w') as fp:
            json.dump(results, fp)
        
        if (retrain == True):
            # Train model on only the selected features
            model = model.fit(train[results[key]], labels)
        return model

    def analyze_sfs(self, data:pd.DataFrame, train:pd.DataFrame, labels_train:pd.Series, 
                    model_dict:dict, agreement:int, direction:str, tree_model:BaseEstimator=None, 
                    evaluate=True, test:pd.DataFrame=None, labels_test:pd.Series=None, 
                    n_features:int|float='auto', tol:float=None, feature_parser:Callable=None,
                    iteration:int=None, save_path:str=None, load_only=False):
        
        if (load_only == False): 
            # Generate new results
            # NOTE: Could run each in parallel, but have opted for a multithreaded sfs
            for name, model in model_dict.items():
                model = self.run_sfs(model, train, labels_train, "{}{}".format(name, iteration), save_path, direction, n_features, tol, evaluate)
                if (evaluate == True):
                    print("{}{} Training results: \n".format(name, iteration))
                    print(classification_report(labels_train, model.predict(train)))
                    if (test is not None and labels_test is not None):
                        print("{}{} Evaluation results: \n".format(name, iteration))
                        print(classification_report(labels_test, model.predict(test)))
            if tree_model is not None:
                # Train and evaluate feature importances for the given tree model
                tree_fitted = tree_model.train(train, labels_train)
                tree_feats = train.columns.to_series().index[pd.Series(tree_fitted.feature_importances_, dtype='bool')].tolist()
                # Save feature importances to save_path
                with open(save_path) as fp:
                    results = json.load(fp)
                results["tree - imp feats{}".format(iteration)] = tree_feats
                with open(save_path, 'w') as fp:
                    json.dump(results, fp)
                if (evaluate == True):
                    print("tree - imp feats{} Training results: \n".format(iteration))
                    print(classification_report(labels_train, tree_fitted.predict(train)))
                    if (test is not None and labels_test is not None):
                        print("tree - imp feats{} Evaluation results: \n".format(iteration))
                        print(classification_report(labels_test, tree_fitted.predict(test)))

        # Load saved results
        with open(save_path) as fp:
            results = json.load(fp)
            if (iteration is not None):
                # Filter the dict by keys for only the cur iteration
                cur_keys = [key for key in results.keys() if str(iteration) in key]
                results = {key: results[key] for key in cur_keys}

        all_feats = {'feats': [], 'model': []}
        for key, value in results.items():
            all_feats['feats'].extend(value)
            all_feats['model'].extend(np.full_like(value, key).tolist())

        # TODO: Save histplot to folder
        plt.figure(figsize=(15,6))
        sns.histplot(data=all_feats, x='feats', stat="count", multiple="stack", element="bars", hue='model', legend=True, shrink=0.5)
        plt.xticks(rotation="vertical")
        plt.show()

        feat_df = pd.DataFrame(all_feats)
        imp_feats = [feat for feat in feat_df['feats'].unique().tolist() if feat_df['feats'].value_counts()[feat] >= agreement]

        sfs_feats = [feat for feat in data.columns.to_list() if any(feat == feature_parser(imp_feat) for imp_feat in imp_feats)]

        return sfs_feats

    def build_feature_chain(self, data:pd.DataFrame, enc_train:pd.DataFrame, labels_train:pd.Series, 
                            model_dict:dict, agreement:int, stages:list, category_dict:str,
                            response:str, direction:str, 
                            enc_test:pd.DataFrame=None, labels_test:pd.Series=None, n_features:int|float='auto', tol:float=None, 
                            evaluate=True, tree_model:BaseEstimator=None, 
                            color_json:str=None, color_swatch:list=None, hover_swatch:list=None,
                            save_dir="./", feature_parser:Callable=None, load_sfs:str=None) -> go.Figure:

        # Load feature categories dict
        with open(category_dict) as fp:
            feature_cats = json.load(fp)

        load_only = load_sfs is not None
        if (load_only == False):
            # Create new file to write results to
            save_path = save_dir + "{}_n{}_{:.0f}_sfs.json".format(direction, n_features, time())
            if tol: save_path = save_dir + "{}_tol{}_{:.0f}_sfs.json".format(direction, tol, time())
            with open(save_path, 'x') as fp:
                # Dump empty dict
                json.dump(dict(), fp)
        else: save_path=load_sfs

        selected_feats = []
        for i, category_list in enumerate(stages):
            cur_feats = []
            for category in category_list:
                cur_feats.extend([feat for feat in enc_train.columns.to_list() if feature_parser(feat) in feature_cats[category]])

            train_data = enc_train[cur_feats]
            test_data = None
            if (enc_test is not None): test_data = enc_test[cur_feats]

            sfs_feats = self.analyze_sfs(data, train_data, labels_train, model_dict,
                                         agreement, direction, tree_model, evaluate, test_data, labels_test,
                                         n_features, tol, feature_parser, i, save_path, load_only)
            
            selected_feats.extend(sfs_feats)

        # Need to append response to plot_feats so it gets plotted
        print(hover_swatch)
        selected_feats.append(response)
        fig = self.build_sankey(data, selected_feats, response, 
                                color_json=color_json, color_swatch=color_swatch, hover_swatch=hover_swatch)
        return fig
