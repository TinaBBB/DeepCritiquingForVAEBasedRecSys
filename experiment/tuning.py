from evaluation.general_performance import evaluate
from prediction.predictor import predict, predict_keyphrase
from utils.io import load_dataframe_csv, save_dataframe_csv, load_yaml
from utils.modelnames import models
from utils.progress import WorkSplitter

import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def hyper_parameter_tuning(train, validation, keyphrase_train, keyphrase_validation, params, save_path, tune_explanation=False):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']

    try:
        df = load_dataframe_csv(table_path, save_path)
    except:
        df = pd.DataFrame(columns=['model', 'rank', 'beta', 'lambda_l2', 'lambda_keyphrase', 'lambda_latent', 'lambda_rating', 'topK', 'learning_rate', 'epoch', 'corruption', 'optimizer'])

    for algorithm in params['models']:

        for rank in params['rank']:

            for beta in params['beta']:

                for lamb_l2 in params['lambda_l2']:

                    for lamb_keyphrase in params['lambda_keyphrase']:

                        for lamb_latent in params['lambda_latent']:

                            for lamb_rating in params['lambda_rating']:

                                for learning_rate in params['learning_rate']:

                                    for epoch in params['epoch']:

                                        for corruption in params['corruption']:

                                            for optimizer in params['optimizer']:

                                                if ((df['model'] == algorithm) &
                                                    (df['rank'] == rank) &
                                                    (df['beta'] == beta) &
                                                    (df['lambda_l2'] == lamb_l2) &
                                                    (df['lambda_keyphrase'] == lamb_keyphrase) &
                                                    (df['lambda_latent'] == lamb_latent) &
                                                    (df['lambda_rating'] == lamb_rating) &
                                                    (df['learning_rate'] == learning_rate) &
                                                    (df['epoch'] == epoch) &
                                                    (df['corruption'] == corruption) &
                                                    (df['optimizer'] == optimizer)).any() or (lamb_latent != lamb_keyphrase):
                                                    continue

                                                format = "model: {}, rank: {}, beta: {}, lambda_l2: {}, " \
                                                    "lambda_keyphrase: {}, lambda_latent: {}, lambda_rating: {}, " \
                                                    "learning_rate: {}, epoch: {}, corruption: {}, optimizer: {}"
                                                progress.section(format.format(algorithm,
                                                                               rank,
                                                                               beta,
                                                                               lamb_l2,
                                                                               lamb_keyphrase,
                                                                               lamb_latent,
                                                                               lamb_rating,
                                                                               learning_rate,
                                                                               epoch,
                                                                               corruption,
                                                                               optimizer))

                                                progress.subsection("Training")

                                                model = models[algorithm](matrix_train=train,
                                                                          epoch=epoch,
                                                                          lamb_l2=lamb_l2,
                                                                          lamb_keyphrase=lamb_keyphrase,
                                                                          lamb_latent=lamb_latent,
                                                                          lamb_rating=lamb_rating,
                                                                          beta=beta,
                                                                          learning_rate=learning_rate,
                                                                          rank=rank,
                                                                          corruption=corruption,
                                                                          optimizer=optimizer,
                                                                          matrix_train_keyphrase=keyphrase_train)

                                                progress.subsection("Prediction")

                                                rating_score, keyphrase_score = model.predict(train.todense())

                                                progress.subsection("Evaluation")

                                                # get explanation results
                                                # if tune_explanation:
                                                uk_prediction = predict_keyphrase(keyphrase_score,
                                                                                  topK=params['topK'][-1])

                                                uk_result = evaluate(uk_prediction,
                                                                  keyphrase_validation,
                                                                  params['metric'],
                                                                  params['topK'])

                                                # get rating prediction results
                                                # else:
                                                ui_prediction = predict(rating_score,
                                                                     topK=params['topK'][-1],
                                                                     matrix_Train=train)

                                                ui_result = evaluate(ui_prediction,
                                                                  validation,
                                                                  params['metric'],
                                                                  params['topK'])

                                                result_dict = {'model': algorithm,
                                                               'rank': rank,
                                                               'beta': beta,
                                                               'lambda_l2': lamb_l2,
                                                               'lambda_keyphrase': lamb_keyphrase,
                                                               'lambda_latent': lamb_latent,
                                                               'lambda_rating': lamb_rating,
                                                               'learning_rate': learning_rate,
                                                               'epoch': epoch,
                                                               'corruption': corruption,
                                                               'optimizer': optimizer}

                                                for name in ui_result.keys():
                                                    result_dict[name] = [round(ui_result[name][0], 4),
                                                                         round(ui_result[name][1], 4)]

                                                for name in uk_result.keys():
                                                    result_dict['uk_' + name] = [round(uk_result[name][0], 4),
                                                                                 round(uk_result[name][1], 4)]

                                                df = df.append(result_dict, ignore_index=True)

                                                model.sess.close()
                                                tf.reset_default_graph()

                                                save_dataframe_csv(df, table_path, save_path)

