from metaflow import FlowSpec, step, IncludeFile

class VictorMetaFlow(FlowSpec):

    # train_data = IncludeFile(
    #     'train_data',
    #     is_text=False,
    #     help='train dataset',
    #     default='/home/vitinleao/ailab/pre-processing-studies/victor/dataset/train_small.csv'
    # )

    # valid_data = IncludeFile(
    #     'valid_data',
    #     is_text=False,
    #     help='validation dataset',
    #     default='/home/vitinleao/ailab/pre-processing-studies/victor/dataset/validation_small.csv'
    # )

    # test_data = IncludeFile(
    #     'test_data',
    #     is_text=False,
    #     help='test dataset',
    #     default='/home/vitinleao/ailab/pre-processing-studies/victor/dataset/test_small.csv'
    # )

    @step
    def start(self):
        self.next(self.load_data)

    @step
    def load_data(self):
        import pandas as pd

        self.train_dataset = pd.read_csv('/home/vitinleao/ailab/pre-processing-studies/victor/dataset/train_small.csv')
        self.valid_dataset = pd.read_csv('/home/vitinleao/ailab/pre-processing-studies/victor/dataset/validation_small.csv')
        self.test_dataset = pd.read_csv('/home/vitinleao/ailab/pre-processing-studies/victor/dataset/test_small.csv')
        
        self.next(self.data_partition)

    @step
    def data_partition(self):
        self.X_train = self.train_dataset['body']
        self.Y_train = self.train_dataset['document_type']

        self.X_valid = self.valid_dataset['body']
        self.Y_valid = self.valid_dataset['document_type']

        self.X_test = self.test_dataset['body']
        self.Y_test = self.test_dataset['document_type']

        self.next(self.gridsearch)

    @step
    def gridsearch(self):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import GridSearchCV

        pipe_countv_sdg = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', SGDClassifier())
        ])

        param_countv_sdg = {
            'vect__min_df': [1],
            'vect__ngram_range': [(1, 3)],
            'vect__max_df': [0.5]
        }

        grid_search_countv_sdg = GridSearchCV(estimator=pipe_countv_sdg, param_grid=param_countv_sdg, scoring='f1_macro', refit='f1_macro')
        grid_search_countv_sdg.fit(self.X_train, self.Y_train)
        self.countv_sdg = grid_search_countv_sdg.best_estimator_

        self.next(self.test)

    @step
    def test(self):
        from sklearn.metrics import f1_score

        print('F1-Score (macro) Test: ', f1_score(self.Y_test, self.countv_sdg.predict(self.X_test), average='macro'))

        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    VictorMetaFlow()