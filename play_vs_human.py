
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR

"""
This file is not runable; it's is a template to show how you could play against your algorithms. To do so,
replace "YourAlgorithmsEvalAgentCls" with the EvalAgent subclass (not instance) of your algorithm.

Note that you can see the AI's cards on the screen since this is just a research application and not meant for actual
competition. The AI can, of course, NOT see your cards.
"""

from PokerRL.game.InteractiveGame import InteractiveGame

if __name__ == '__main__':
    eval_agent = EvalAgentDeepCFR.load_from_disk(
        path_to_eval_agent="/home/stefan/poker_ai_data/eval_agent/SD-CFR_LIMIT/20/eval_agentSINGLE.pkl")

    game = InteractiveGame(env_cls=eval_agent.env_bldr.env_cls,
                           env_args=eval_agent.env_bldr.env_args,
                           seats_human_plays_list=[0],
                           eval_agent=eval_agent,
                           )

    game.start_to_play()
