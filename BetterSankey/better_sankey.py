import json
import scipy
import time
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
        pass

    def build_sankey(self, data:pd.DataFrame, feats:list, response:str, 
                     significance=True, color_json:str=None,
                     color_swatch:list=None, hover_swatch:list=None, vertical_pad=15) -> go.Figure:
        labels = []
        sources = []
        targets = []
        values = []
        x_pos = []
        y_pos = []
        p_values = []
        link_colors = []
        link_labels = []
        hover_colors = []
        response_vals = list(data[response].value_counts(dropna=False).sort_values(ascending=False).index)

        val_counts = []
        x_reference = np.linspace(0.0, 1.0, num=len(feats)).tolist()
        for i, feat in enumerate(feats):
            val_counts.append(data[feat].value_counts(dropna=False).sort_values(ascending=False))
            unique_vals = list(val_counts[i].index)
            labels.extend([feat + ': ' + str(val) for val in unique_vals])
            x_pos.extend(np.full(len(unique_vals), x_reference[i]))
            # Accurately calculate y-position between 0 and 1
            level_counts = val_counts[i].to_list()
            level_props = [count / sum(level_counts) for count in level_counts]
            y_align = [1 - (sum(level_props[:i]) + (val / 2)) for i, val in enumerate(level_props)]
            # Reverse list
            y_pos.extend(y_align[::-1])
            # Do chi^2 tests
            if (feat != response):
                contingency_tab = pd.crosstab(data[feat], data[response], margins=False, dropna=False)
                x2_res = scipy.stats.chi2_contingency(contingency_tab)
                p_values.append(x2_res.pvalue)
            if (i > 0):
                prior_unique_vals = list(val_counts[i - 1].index)
                for j, value in enumerate(unique_vals):
                    for k, prior_value in enumerate(prior_unique_vals):
                        for x, response_value in enumerate(response_vals):
                            # Count num samples with both values
                            if pd.isnull(value): 
                                value_mask = data[feat].isna()
                            else: value_mask = data[feat] == value
                            
                            if pd.isnull(prior_value): 
                                prior_value_mask = data[feats[i-1]].isna()
                            else: prior_value_mask = data[feats[i-1]] == prior_value

                            if pd.isnull(response_value): 
                                response_value_mask = data[response].isna()
                            else: response_value_mask = data[response] == response_value

                            values.append(len(data[(value_mask) & (prior_value_mask) & (response_value_mask)]) / len(data))
                            # Set source: index labels by -1 * (len(val_counts) + len(prior_val_counts) - k)
                            sources.append(labels.index(labels[-1 * (len(unique_vals) + len(prior_unique_vals) - k)]))
                            # Set target: index labels by -1 * (len(val_counts) - j)
                            targets.append(labels.index(labels[-1 * (len(unique_vals) - j)]))
                            link_colors.append(color_swatch[x])
                            link_labels.append(response + ": " + str(response_value))
                            hover_colors.append(hover_swatch[x])
        
        # Create sankey/alluvial diagram
        colors = None
        if (color_json is not None): 
            # Load color dict
            with open(color_json) as fp:
                color_dict = json.load(fp)

            colors = [color_dict[label.split(':')[0]] for label in labels[:-len(response_vals)]]
            colors.extend(hover_swatch[:len(response_vals)])

        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            valueformat = ".2%",
            node = dict(
                pad = vertical_pad,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = labels,
                color = colors,
                x = x_pos,
                y = y_pos,
            ),
            link = dict(
                source = sources, # indices correspond to labels
                target = targets,
                value = values,
                label = link_labels,
                color = link_colors,
                # BUG: In Plotly package, does not update colors
                hovercolor = hover_colors,
            ))])

        if (significance):
            # Add chi^2 p-vals as annotations
            # NOTE: Assumes response is last (x_reference[-1])
            for i, p in enumerate(p_values):
                fig.add_annotation(
                    text = "p-value: {:.3e}".format(p),
                    x = x_reference[i],
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

        plt.figure(figsize=(15,6))
        sns.histplot(data=all_feats, x='feats', stat="count", multiple="stack", element="bars", hue='model', legend=True, shrink=0.5)
        plt.xticks(rotation="vertical")
        plt.show()

        feat_df = pd.DataFrame(all_feats)
        imp_feats = [feat for feat in feat_df['feats'].unique().tolist() if feat_df['feats'].value_counts()[feat] >= agreement]

        sfs_feats = [feat for feat in data.columns.to_list() if any(feat == feature_parser(imp_feat) for imp_feat in imp_feats)]

        return sfs_feats

    def build_feature_chain(self, data:pd.DataFrame, enc_train:pd.DataFrame, labels_train:pd.Series, 
                            enc_test:pd.DataFrame, labels_test:pd.Series, agreement:int, stages:list, category_dict:str,
                            response:str, direction:str, n_features:int|float='auto', tol:float=None, 
                            feature_parser:Callable=None, saved_path:str=None):

        # Load feature categories dict
        with open(category_dict) as fp:
            feature_cats = json.load(fp)

        load_only = saved_path is not None
        if (load_only == False):
            # Create new file to write results to
            saved_path = './{}_n{}_{:.0f}_sfs.json'.format(direction, n_features, time())
            if tol: saved_path = './{}_tol{}_{:.0f}_sfs.json'.format(direction, tol, time())
            with open(saved_path, 'x') as fp:
                # Dump empty dict
                json.dump(dict(), fp)

        selected_feats = []
        for i, category_list in enumerate(stages):
            cur_feats = []
            for category in category_list:
                cur_feats.extend([feat for feat in enc_train.columns.to_list() if feature_parser(feat) in feature_cats[category]])

            train_data = enc_train[cur_feats]
            test_data = enc_test[cur_feats]

            sfs_feats = self.analyze_sfs(data, train_data, labels_train, test_data, labels_test,
                                    agreement, direction, n_features, tol, 
                                    i, saved_path, load_only)
            selected_feats.extend(sfs_feats)

        # Need to append response to plot_feats so it gets plotted
        selected_feats.append(response)
        self.build_sankey(data, selected_feats, response)
