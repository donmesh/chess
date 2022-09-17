import os
import pickle
import argparse
from configs.paths import SCORES_2014, SCORES_2020, TEST_DATA_PATH, TRAIN_DATA_PATH, MODEL_DIR
from utils.name_mapping import create_name_mapping
from tournaments.data_cleaning import get_all_tournament_data
from scores.data_cleaning import get_combined_scores

if __name__ == '__main__':
      argparser  = argparse.ArgumentParser()
      argparser.add_argument('--scores-2014', type=str, default=SCORES_2014, help='Path to ELO 2014 scores')
      argparser.add_argument('--scores-2020', type=str, default=SCORES_2020, help='Path to ELO 2020 scores')
      argparser.add_argument('--tournament-train', type=str, default=TRAIN_DATA_PATH, help='Path to tournament training data')
      argparser.add_argument('--tournament-test', type=str, default=TEST_DATA_PATH, help='Path to tournament testing data')

      args = argparser.parse_args()

      scores = get_combined_scores(args.scores_2014, args.scores_2020)
      tournaments_train = get_all_tournament_data(args.tournament_train, True)
      tournaments_test = get_all_tournament_data(args.tournament_test, True)
      names = tournaments_train.reindex(columns=['white', 'black']).stack().unique().tolist() \
            + tournaments_test.reindex(columns=['white', 'black']).stack().unique().tolist() \
            + scores.index.tolist()
      name_mapping = create_name_mapping(names)

      os.makedirs(MODEL_DIR, exist_ok=True)
      with open(f'{MODEL_DIR}/name_mapping.pickle', 'wb') as f:
            pickle.dump(name_mapping, f)