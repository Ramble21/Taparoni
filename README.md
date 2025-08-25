Simple chess engine made from the ground up completely by me, for fun. WIP, not complete yet

More information about the model: <br>
- Uses a custom-built Transformer to analyze positions, using Stockfish evaluations as references to train the model
- Able to play moves by iterating through all possible legal moves and evaluating the positions if they were to be played, with variable depth. Functionality for this part is still WIP and buggy.
- Trained on public data from https://database.lichess.org/ 

Current model stats:
- Games trained on: 175,033
- Positions trained on: 12,082,774
- Finetuning L1 loss: 0.1015

More capabilities coming soon <br>
Some more documentation on the building of the model: https://lichess.org/study/20AlyXhA
