from PokerRL.game.games import LimitHoldem  # or any other game

from PokerRL.eval.rl_br import RLBRArgs
from PokerRL.eval.lbr import LBRArgs
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver
from PokerRL.game import Poker

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(name="SD-CFR_LIMIT",
                                         nn_type="feedforward",
                                         max_buffer_size_adv=3e6,
                                         eval_agent_export_freq=20,  # export API to play against the agent
                                         n_traversals_per_iter=1500,
                                         n_batches_adv_training=4000,
                                         n_batches_avrg_training=20000,
                                         n_merge_and_table_layer_units_adv=64,
                                         n_merge_and_table_layer_units_avrg=64,
                                         n_units_final_adv=64,
                                         n_units_final_avrg=64,
                                         mini_batch_size_adv=2048,
                                         mini_batch_size_avrg=2048,
                                         init_adv_model="last",
                                         init_avrg_model="last",
                                         use_pre_layers_adv=False,
                                         use_pre_layers_avrg=False,
                                         n_seats=2,
                                         game_cls=LimitHoldem,

                                         device_training='cuda',
                                         device_inference='cuda',
                                         device_parameter_server='cuda',

                                         # You can specify one or both modes. Choosing both is useful to compare them.
                                         eval_modes_of_algo=[
                                             EvalAgentDeepCFR.EVAL_MODE_SINGLE  # SD-CFR
                                         ],

                                         DISTRIBUTED=True,
                                         n_learner_actor_workers=10,

                                         # rl_br_args=RLBRArgs(
                                         #     n_workers=4,
                                         #     device_training='cuda',
                                         #     rlbr_bet_set=1
                                         # ),
                                         lbr_args=LBRArgs(
                                            n_parallel_lbr_workers=10,
                                            use_gpu_for_batch_eval=True,
                                            DISTRIBUTED=True,
                                            lbr_check_to_round=Poker.TURN
                                         )

                                         ),


                  eval_methods={
                      "lbr": 15,
                  },
                  n_iterations=None)
    ctrl.run()
