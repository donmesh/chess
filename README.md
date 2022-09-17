# Summary

## Assumptions
- maximum score is taken if there are multiple scores per person
- players with missing scores are assigned a score of 0

## Limitations
- Names are not checked for spelling errors
- Historical games between the same opponents are not explicitly accounted for
    - Data is not treated as time series
    - Data is not treated as sequence of games
- Neither of the current features show promissing correlation with the results
- Future events have impact on past events (e.g., score_2020 on games played in 2014)

## Excluded features:
- tournament_id: id column
- game_id: id column

## Created features:
- white: player names are mapped to generic ids
- black: player names are mapped to generic ids
- is_classic: 1 if classic tournament type, 0 otherwise
- is_knockout: derived based on number of players and tours; 1 if knockout type, 0 otherwise
    - is_round_robin: always 0, thus, excluded
    - is_swiss: opposite to knockout, excluded
- tour_completion: ratio of the current tour and total tours in the tournament
- tournament_completion: ratio of days since tournament start and total allocated days
- white_score_2014: rating of white player according ELO 2014
- white_score_2020: rating of white player according ELO 2020
- black_score_2014: rating of black player according ELO 2014
- black_score_2020: rating of black player according ELO 2020

# Project directory
```bash
├───data
│   ├───test
│   ├───train
├───model_files
├───notebooks
├───output_files
├───scripts
└───src
    ├───api
    ├───configs
    ├───models
    │   ├───advanced_models
    │   ├───baseline_models
    │   ├───operations
    ├───plotting
    ├───postprocessing
    ├───preprocessing
    ├───scores
    ├───tests
    ├───tools
    ├───tournaments
    └───utils
```

# Set up
1. set HOST and PORT values in docker-compose.yaml
2. build Docker image: `docker-compose build`
3. run Docker image: `docker-compose up`
4. if there is no `model_files/name_mapping.pickle`:
    1) login into the container: `docker exec -it chess bash`
    2) run script to create the missing file: `python tools/create_name_mapping.py`
5. interact with application using dedicated endpoints

# Documentation
Available endpoints and SwaggerUI: `HOST:PORT/docs`