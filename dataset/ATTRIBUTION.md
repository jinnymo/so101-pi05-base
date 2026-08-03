# Attribution

This dataset redistributes work created by other people. This file is the attribution notice required by Section 4 of the Apache License 2.0, and it records the provenance of every episode in the release.

- 156 source datasets, 13,969 episodes
- apache-2.0, from the Hugging Face Hub: 149 datasets, 13,587 episodes
- mit, from the Hugging Face Hub: 1 dataset, 50 episodes
- apache-2.0, recorded by Dongyoon Kim: 6 datasets, 332 episodes

The episode count is the number of episodes taken from that source after deduplication and per-episode drops. It can be lower than the episode count of the upstream repository. Licenses were read from the upstream repository metadata at collection time; upstream repositories can change, and the upstream entry is authoritative.

## Licenses

A copy of the Apache License 2.0 is distributed with this dataset as `LICENSE`, as Section 4(a) of that license requires.

The six sources recorded by Dongyoon Kim are released under the Apache License 2.0. "Recorded by the author" is a provenance category, not a license.

One source, [yuk6ra/so101-pen-cleanup](https://huggingface.co/datasets/yuk6ra/so101-pen-cleanup) (50 episodes), is MIT rather than apache-2.0. MIT permits use, copying, modification and redistribution provided its copyright notice and permission notice are included in all copies. The upstream repository declares `license: mit` in its dataset card metadata and ships no `LICENSE` file and no copyright line, so there is no upstream notice text to reproduce verbatim; this entry and the table row below are the attribution. The MIT permission notice, in its standard form, reads:

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Modifications

Section 4(b) of the Apache License 2.0 requires stating that files were changed. Every source dataset was modified as follows before inclusion:

- datasets published in LeRobot v3.0 format were converted to v2.1
- camera streams were remapped onto three fixed slots (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`); a slot left empty by the keyword pass was filled from whatever cameras remained, slots still empty after that were filled with black placeholder video, and a per-slot mask column was added
- depth, infrared and surplus camera streams were dropped
- episode, frame and task indices were renumbered into a single global index space, and the per-source metadata files were regenerated
- a small number of task strings were rewritten into plain English instructions (integer placeholders, dict reprs, snake_case, one non-English source)
- individual episodes were dropped where they were empty, truncated or corrupt

No action, state or image content was otherwise altered.

## External sources

| Source | License | Episodes |
|---|---|---|
| [5hadytru/so101_grasp_1](https://huggingface.co/datasets/5hadytru/so101_grasp_1) | apache-2.0 | 210 |
| [5hadytru/so101_grasp_2](https://huggingface.co/datasets/5hadytru/so101_grasp_2) | apache-2.0 | 780 |
| [5hadytru/so101_grasp_3](https://huggingface.co/datasets/5hadytru/so101_grasp_3) | apache-2.0 | 370 |
| [5hadytru/so101_IF_1](https://huggingface.co/datasets/5hadytru/so101_IF_1) | apache-2.0 | 82 |
| [5hadytru/so101_IF_2](https://huggingface.co/datasets/5hadytru/so101_IF_2) | apache-2.0 | 78 |
| [5hadytru/so101_IF_3](https://huggingface.co/datasets/5hadytru/so101_IF_3) | apache-2.0 | 67 |
| [aaron-ser/so101-dice-pickup-dataset](https://huggingface.co/datasets/aaron-ser/so101-dice-pickup-dataset) | apache-2.0 | 70 |
| [aaronsu11/so101_fruit](https://huggingface.co/datasets/aaronsu11/so101_fruit) | apache-2.0 | 200 |
| [aaronsu11/so101_fruit_leisaac](https://huggingface.co/datasets/aaronsu11/so101_fruit_leisaac) | apache-2.0 | 57 |
| [aiden-li/so101-close-lower-drawer](https://huggingface.co/datasets/aiden-li/so101-close-lower-drawer) | apache-2.0 | 59 |
| [aiden-li/so101-close-upper-drawer](https://huggingface.co/datasets/aiden-li/so101-close-upper-drawer) | apache-2.0 | 150 |
| [aiden-li/so101-grabtissue](https://huggingface.co/datasets/aiden-li/so101-grabtissue) | apache-2.0 | 144 |
| [aiden-li/so101-open-lower-drawer](https://huggingface.co/datasets/aiden-li/so101-open-lower-drawer) | apache-2.0 | 144 |
| [aiden-li/so101-open-upper-drawer](https://huggingface.co/datasets/aiden-li/so101-open-upper-drawer) | apache-2.0 | 154 |
| [aiden-li/so101-picklego](https://huggingface.co/datasets/aiden-li/so101-picklego) | apache-2.0 | 113 |
| [aiden-li/so101-picktape](https://huggingface.co/datasets/aiden-li/so101-picktape) | apache-2.0 | 159 |
| [ankithreddy/so101_pickplace_tools](https://huggingface.co/datasets/ankithreddy/so101_pickplace_tools) | apache-2.0 | 90 |
| [ankithreddy/so101_pickplace_uno](https://huggingface.co/datasets/ankithreddy/so101_pickplace_uno) | apache-2.0 | 50 |
| [anvilbot-patrickhhh/SO101_PickAndPlace_3cams](https://huggingface.co/datasets/anvilbot-patrickhhh/SO101_PickAndPlace_3cams) | apache-2.0 | 50 |
| [anvilbot-patrickhhh/SO101_PickAndPlace_front_wrist](https://huggingface.co/datasets/anvilbot-patrickhhh/SO101_PickAndPlace_front_wrist) | apache-2.0 | 50 |
| [anvilbot-patrickhhh/SO101_record_test](https://huggingface.co/datasets/anvilbot-patrickhhh/SO101_record_test) | apache-2.0 | 100 |
| [anvilbot-patrickhhh/SO101_relocate_cube_2cams_record_2](https://huggingface.co/datasets/anvilbot-patrickhhh/SO101_relocate_cube_2cams_record_2) | apache-2.0 | 100 |
| [azaracla/so101_pick_3dprint](https://huggingface.co/datasets/azaracla/so101_pick_3dprint) | apache-2.0 | 50 |
| [baluatbfl/my-so101arm-dataset](https://huggingface.co/datasets/baluatbfl/my-so101arm-dataset) | apache-2.0 | 50 |
| [Benxiaogu/SO101-full](https://huggingface.co/datasets/Benxiaogu/SO101-full) | apache-2.0 | 100 |
| [brcg3/so101_test](https://huggingface.co/datasets/brcg3/so101_test) | apache-2.0 | 53 |
| [brcg3/so101_test_2](https://huggingface.co/datasets/brcg3/so101_test_2) | apache-2.0 | 52 |
| [c299m/so101-pen-in-box-v2](https://huggingface.co/datasets/c299m/so101-pen-in-box-v2) | apache-2.0 | 130 |
| [ceva-automation-sg/smolVLA_so101](https://huggingface.co/datasets/ceva-automation-sg/smolVLA_so101) | apache-2.0 | 90 |
| [CnLori/so101_grab_cube](https://huggingface.co/datasets/CnLori/so101_grab_cube) | apache-2.0 | 50 |
| [CnLori/so101_pickup_cube](https://huggingface.co/datasets/CnLori/so101_pickup_cube) | apache-2.0 | 56 |
| [CoRL2026-CSI/IsaacLab-SO101-PullCube-100epi-10fps-appendix](https://huggingface.co/datasets/CoRL2026-CSI/IsaacLab-SO101-PullCube-100epi-10fps-appendix) | apache-2.0 | 100 |
| [CoRL2026-CSI/SO101-teleop_stack_RGBblock_on_bluedish_150epi_10fps](https://huggingface.co/datasets/CoRL2026-CSI/SO101-teleop_stack_RGBblock_on_bluedish_150epi_10fps) | apache-2.0 | 150 |
| [CrawlAiHuggingFace/multiplied_so101-table-cleanup](https://huggingface.co/datasets/CrawlAiHuggingFace/multiplied_so101-table-cleanup) | apache-2.0 | 79 |
| [Damin3927/so101_pickplace](https://huggingface.co/datasets/Damin3927/so101_pickplace) | apache-2.0 | 50 |
| [DeanSu948/so101_test_0815](https://huggingface.co/datasets/DeanSu948/so101_test_0815) | apache-2.0 | 50 |
| [dleon23/record-so101_2](https://huggingface.co/datasets/dleon23/record-so101_2) | apache-2.0 | 101 |
| [EverNorif/so101-table-cleanup](https://huggingface.co/datasets/EverNorif/so101-table-cleanup) | apache-2.0 | 82 |
| [EverNorif/so101_place_orange_isaaclab](https://huggingface.co/datasets/EverNorif/so101_place_orange_isaaclab) | apache-2.0 | 83 |
| [fbeltrao/so101_multi_task_v2](https://huggingface.co/datasets/fbeltrao/so101_multi_task_v2) | apache-2.0 | 106 |
| [fuemocheng/record_so101_mahjong_test01](https://huggingface.co/datasets/fuemocheng/record_so101_mahjong_test01) | apache-2.0 | 60 |
| [funXedu/so101_lego_brick](https://huggingface.co/datasets/funXedu/so101_lego_brick) | apache-2.0 | 86 |
| [geekscape/lerobot_so101_test_00](https://huggingface.co/datasets/geekscape/lerobot_so101_test_00) | apache-2.0 | 50 |
| [gimarchetti/so101-winnie-us5](https://huggingface.co/datasets/gimarchetti/so101-winnie-us5) | apache-2.0 | 53 |
| [gimarchetti/so101-winnie-us7](https://huggingface.co/datasets/gimarchetti/so101-winnie-us7) | apache-2.0 | 121 |
| [Grigorij/so-101-duck](https://huggingface.co/datasets/Grigorij/so-101-duck) | apache-2.0 | 70 |
| [Grigorij/so-101-test](https://huggingface.co/datasets/Grigorij/so-101-test) | apache-2.0 | 50 |
| [guanfengliu/so101_main_bin_3cameras_1depth](https://huggingface.co/datasets/guanfengliu/so101_main_bin_3cameras_1depth) | apache-2.0 | 62 |
| [guanfengliu/so101_main_bin_3cameras_3](https://huggingface.co/datasets/guanfengliu/so101_main_bin_3cameras_3) | apache-2.0 | 55 |
| [guanfengliu/so101_main_bin_yellow4](https://huggingface.co/datasets/guanfengliu/so101_main_bin_yellow4) | apache-2.0 | 100 |
| [Jaehooni/smolvla_so101_pick_and_place](https://huggingface.co/datasets/Jaehooni/smolvla_so101_pick_and_place) | apache-2.0 | 80 |
| [JiabinQ/so101_pick_place](https://huggingface.co/datasets/JiabinQ/so101_pick_place) | apache-2.0 | 50 |
| [joadv/so101-test](https://huggingface.co/datasets/joadv/so101-test) | apache-2.0 | 50 |
| [jrkhf/so101_set_2](https://huggingface.co/datasets/jrkhf/so101_set_2) | apache-2.0 | 50 |
| [jrkhf/so101_wrist_top_cameras_set_1](https://huggingface.co/datasets/jrkhf/so101_wrist_top_cameras_set_1) | apache-2.0 | 50 |
| [jrkhf/so101_wrist_top_cameras_set_2](https://huggingface.co/datasets/jrkhf/so101_wrist_top_cameras_set_2) | apache-2.0 | 103 |
| [JulienStocker/so101-cup_pnp_50ep_ver01](https://huggingface.co/datasets/JulienStocker/so101-cup_pnp_50ep_ver01) | apache-2.0 | 50 |
| [k1000dai/so101_put_on_conveyor_mix_training-smolvla](https://huggingface.co/datasets/k1000dai/so101_put_on_conveyor_mix_training-smolvla) | apache-2.0 | 200 |
| [kagyvro48/so101_dataset1_arracher_la_mauvaise_herbe](https://huggingface.co/datasets/kagyvro48/so101_dataset1_arracher_la_mauvaise_herbe) | apache-2.0 | 61 |
| [Kazu1232/record-so101-A_warp50](https://huggingface.co/datasets/Kazu1232/record-so101-A_warp50) | apache-2.0 | 50 |
| [Kazu1232/record-so101-B_warp50](https://huggingface.co/datasets/Kazu1232/record-so101-B_warp50) | apache-2.0 | 50 |
| [Kazu1232/record-so101-C_warp50](https://huggingface.co/datasets/Kazu1232/record-so101-C_warp50) | apache-2.0 | 50 |
| [Kazu1232/record-so101-warp-ABC-reverse_A50](https://huggingface.co/datasets/Kazu1232/record-so101-warp-ABC-reverse_A50) | apache-2.0 | 50 |
| [Kazu1232/record-so101-warp-ABC-reverse_B50](https://huggingface.co/datasets/Kazu1232/record-so101-warp-ABC-reverse_B50) | apache-2.0 | 50 |
| [Kazu1232/record-so101-warp-ABC-reverse_C50](https://huggingface.co/datasets/Kazu1232/record-so101-warp-ABC-reverse_C50) | apache-2.0 | 50 |
| [leesangoh/so101-pick-and-place-red-bus](https://huggingface.co/datasets/leesangoh/so101-pick-and-place-red-bus) | apache-2.0 | 50 |
| [LeRobot-worldwide-hackathon/241-Sushi_Shinkansen_So101-pick_sushi](https://huggingface.co/datasets/LeRobot-worldwide-hackathon/241-Sushi_Shinkansen_So101-pick_sushi) | apache-2.0 | 147 |
| [LightwheelAI/so101-pick-pen](https://huggingface.co/datasets/LightwheelAI/so101-pick-pen) | apache-2.0 | 61 |
| [LightwheelAI/so101-place-orange](https://huggingface.co/datasets/LightwheelAI/so101-place-orange) | apache-2.0 | 61 |
| [lipsop/so101-block-in-bin-100ep](https://huggingface.co/datasets/lipsop/so101-block-in-bin-100ep) | apache-2.0 | 100 |
| [littledragon/so101_sock_stowing2](https://huggingface.co/datasets/littledragon/so101_sock_stowing2) | apache-2.0 | 59 |
| [LittleFire99/so101_test_1](https://huggingface.co/datasets/LittleFire99/so101_test_1) | apache-2.0 | 90 |
| [lucasfv/so101_finetuning](https://huggingface.co/datasets/lucasfv/so101_finetuning) | apache-2.0 | 60 |
| [MCeut/so101_lab_openBin_putCube](https://huggingface.co/datasets/MCeut/so101_lab_openBin_putCube) | apache-2.0 | 103 |
| [mr-dee/dylan-so101-train1](https://huggingface.co/datasets/mr-dee/dylan-so101-train1) | apache-2.0 | 50 |
| [nbirukov/so101_pick_stack_ring_pole](https://huggingface.co/datasets/nbirukov/so101_pick_stack_ring_pole) | apache-2.0 | 100 |
| [nbirukov/so101_pick_up_3c_zone_2](https://huggingface.co/datasets/nbirukov/so101_pick_up_3c_zone_2) | apache-2.0 | 130 |
| [ndelgado/so101_dataset](https://huggingface.co/datasets/ndelgado/so101_dataset) | apache-2.0 | 50 |
| [nikodembartnik/so101-task1](https://huggingface.co/datasets/nikodembartnik/so101-task1) | apache-2.0 | 50 |
| [observabot/so101_die_mat4](https://huggingface.co/datasets/observabot/so101_die_mat4) | apache-2.0 | 217 |
| [omkarmayekar555/sim_so101_follower_14_OCT_1_6_loc_one_orange](https://huggingface.co/datasets/omkarmayekar555/sim_so101_follower_14_OCT_1_6_loc_one_orange) | apache-2.0 | 64 |
| [omkarmayekar555/so101_test_dataset2](https://huggingface.co/datasets/omkarmayekar555/so101_test_dataset2) | apache-2.0 | 50 |
| [oretti/so101_dice_5](https://huggingface.co/datasets/oretti/so101_dice_5) | apache-2.0 | 60 |
| [peterrolfes/so101_grab_the_screw_big](https://huggingface.co/datasets/peterrolfes/so101_grab_the_screw_big) | apache-2.0 | 126 |
| [piuslim373/so101-transfer-bottle](https://huggingface.co/datasets/piuslim373/so101-transfer-bottle) | apache-2.0 | 50 |
| [piuslim373/so101-transfer-bottle1](https://huggingface.co/datasets/piuslim373/so101-transfer-bottle1) | apache-2.0 | 50 |
| [piuslim373/so101-transfer-capsule](https://huggingface.co/datasets/piuslim373/so101-transfer-capsule) | apache-2.0 | 101 |
| [piuslim373/so101-transfer-capsule3](https://huggingface.co/datasets/piuslim373/so101-transfer-capsule3) | apache-2.0 | 50 |
| [piuslim373/so101-transfer-capsule4](https://huggingface.co/datasets/piuslim373/so101-transfer-capsule4) | apache-2.0 | 50 |
| [piuslim373/so101-transfer-capsule5](https://huggingface.co/datasets/piuslim373/so101-transfer-capsule5) | apache-2.0 | 80 |
| [puneetpanwar/so101-table_cleanup](https://huggingface.co/datasets/puneetpanwar/so101-table_cleanup) | apache-2.0 | 50 |
| [puneetpanwar/so101_cube_pickup](https://huggingface.co/datasets/puneetpanwar/so101_cube_pickup) | apache-2.0 | 100 |
| [qm30631122/so101_grab_pen_blocked_20sec_50ep_102425](https://huggingface.co/datasets/qm30631122/so101_grab_pen_blocked_20sec_50ep_102425) | apache-2.0 | 50 |
| [r-oi/so101_pickplace_cube_100ep_rand-pos](https://huggingface.co/datasets/r-oi/so101_pickplace_cube_100ep_rand-pos) | apache-2.0 | 100 |
| [r-oi/so101_pickplace_cube_1020_same_place](https://huggingface.co/datasets/r-oi/so101_pickplace_cube_1020_same_place) | apache-2.0 | 50 |
| [r-oi/so101_pickplace_cube_80ep](https://huggingface.co/datasets/r-oi/so101_pickplace_cube_80ep) | apache-2.0 | 80 |
| [r2owb0/so101-DS1](https://huggingface.co/datasets/r2owb0/so101-DS1) | apache-2.0 | 51 |
| [ranegray/so101-drawing-dataset](https://huggingface.co/datasets/ranegray/so101-drawing-dataset) | apache-2.0 | 51 |
| [ricky0526/so101_move_block](https://huggingface.co/datasets/ricky0526/so101_move_block) | apache-2.0 | 120 |
| [ricky0526/so101_pick_mouse_to_box_v1](https://huggingface.co/datasets/ricky0526/so101_pick_mouse_to_box_v1) | apache-2.0 | 60 |
| [ricky0526/so101_pick_toy_to_plate_v1](https://huggingface.co/datasets/ricky0526/so101_pick_toy_to_plate_v1) | apache-2.0 | 50 |
| [ricky0526/so101_pick_toy_to_plate_v2](https://huggingface.co/datasets/ricky0526/so101_pick_toy_to_plate_v2) | apache-2.0 | 50 |
| [ricky0526/so101_pick_toy_to_plate_v3](https://huggingface.co/datasets/ricky0526/so101_pick_toy_to_plate_v3) | apache-2.0 | 80 |
| [ricky0526/so101_pick_toy_to_plate_v4](https://huggingface.co/datasets/ricky0526/so101_pick_toy_to_plate_v4) | apache-2.0 | 80 |
| [rl26-world-models/so101-task2-720p-whole-arm-v4-fresh](https://huggingface.co/datasets/rl26-world-models/so101-task2-720p-whole-arm-v4-fresh) | apache-2.0 | 50 |
| [rowb1/so101_pick_cup1](https://huggingface.co/datasets/rowb1/so101_pick_cup1) | apache-2.0 | 50 |
| [rowb1/so101_pick_cup2](https://huggingface.co/datasets/rowb1/so101_pick_cup2) | apache-2.0 | 50 |
| [sabinMlminator/so101_pickplace_cubes_test1](https://huggingface.co/datasets/sabinMlminator/so101_pickplace_cubes_test1) | apache-2.0 | 200 |
| [sabinMlminator/so101_pickplace_cubes_test2](https://huggingface.co/datasets/sabinMlminator/so101_pickplace_cubes_test2) | apache-2.0 | 120 |
| [Seungyoun/so101-pick-and-place](https://huggingface.co/datasets/Seungyoun/so101-pick-and-place) | apache-2.0 | 121 |
| [shimazukosen/so101_record_2](https://huggingface.co/datasets/shimazukosen/so101_record_2) | apache-2.0 | 80 |
| [SurajChess/so101-left-arm-pick-green-gear](https://huggingface.co/datasets/SurajChess/so101-left-arm-pick-green-gear) | apache-2.0 | 50 |
| [SurajChess/so101-left-arm-pick-grey-gear](https://huggingface.co/datasets/SurajChess/so101-left-arm-pick-grey-gear) | apache-2.0 | 50 |
| [SurajChess/so101-left-arm-pick-red-gear](https://huggingface.co/datasets/SurajChess/so101-left-arm-pick-red-gear) | apache-2.0 | 50 |
| [SurajChess/so101-left-arm-pick-white-object](https://huggingface.co/datasets/SurajChess/so101-left-arm-pick-white-object) | apache-2.0 | 50 |
| [SurajChess/so101-leftarm-red-gear-pickshowplace_brownsmallyellow](https://huggingface.co/datasets/SurajChess/so101-leftarm-red-gear-pickshowplace_brownsmallyellow) | apache-2.0 | 50 |
| [SurajChess/so101-leftarm-red-gear-pickshowplace_greyyellow](https://huggingface.co/datasets/SurajChess/so101-leftarm-red-gear-pickshowplace_greyyellow) | apache-2.0 | 50 |
| [SurajChess/so101-leftarm-red-gear-pickshowplace_greyyellowbrownsmallyellow](https://huggingface.co/datasets/SurajChess/so101-leftarm-red-gear-pickshowplace_greyyellowbrownsmallyellow) | apache-2.0 | 50 |
| [SurajChess/so101-leftarm-yellow-gear-pickshowplace_brownsmallyellow](https://huggingface.co/datasets/SurajChess/so101-leftarm-yellow-gear-pickshowplace_brownsmallyellow) | apache-2.0 | 50 |
| [SurajChess/so101-leftarm-yellow-gear-pickshowplace_redgreen](https://huggingface.co/datasets/SurajChess/so101-leftarm-yellow-gear-pickshowplace_redgreen) | apache-2.0 | 50 |
| [SurajChess/so101-leftarm-yellow-gear-pickshowplace_redgreenbrownsmallyellow](https://huggingface.co/datasets/SurajChess/so101-leftarm-yellow-gear-pickshowplace_redgreenbrownsmallyellow) | apache-2.0 | 50 |
| [SurajChess/so101_dataset3](https://huggingface.co/datasets/SurajChess/so101_dataset3) | apache-2.0 | 50 |
| [SurajChess/so101_dataset_marker_tape_1](https://huggingface.co/datasets/SurajChess/so101_dataset_marker_tape_1) | apache-2.0 | 160 |
| [SurajChess/so101_dataset_tape](https://huggingface.co/datasets/SurajChess/so101_dataset_tape) | apache-2.0 | 51 |
| [Teddy14/so101_test](https://huggingface.co/datasets/Teddy14/so101_test) | apache-2.0 | 90 |
| [Teddy14/so101_two_cam](https://huggingface.co/datasets/Teddy14/so101_two_cam) | apache-2.0 | 90 |
| [Temmp1e/so101_cleanup](https://huggingface.co/datasets/Temmp1e/so101_cleanup) | apache-2.0 | 75 |
| [Thytu/so101-object-in-box_v0.4-fixed](https://huggingface.co/datasets/Thytu/so101-object-in-box_v0.4-fixed) | apache-2.0 | 101 |
| [tinkhireeva/so101_pick_place_yellow_objects](https://huggingface.co/datasets/tinkhireeva/so101_pick_place_yellow_objects) | apache-2.0 | 60 |
| [tinkhireeva/so101_pick_place_yellow_objects_10_locations_dima](https://huggingface.co/datasets/tinkhireeva/so101_pick_place_yellow_objects_10_locations_dima) | apache-2.0 | 50 |
| [TzuShian/so101_grabchess_250924](https://huggingface.co/datasets/TzuShian/so101_grabchess_250924) | apache-2.0 | 120 |
| [TzuShian/so101_white_chess_20251021](https://huggingface.co/datasets/TzuShian/so101_white_chess_20251021) | apache-2.0 | 274 |
| [un1c0rnio/so101_eraser_mat1](https://huggingface.co/datasets/un1c0rnio/so101_eraser_mat1) | apache-2.0 | 50 |
| [wantobcm/so101_box2bowl](https://huggingface.co/datasets/wantobcm/so101_box2bowl) | apache-2.0 | 162 |
| [y1y2y3/so101_test8](https://huggingface.co/datasets/y1y2y3/so101_test8) | apache-2.0 | 250 |
| [YDY0427/so101_test_20250708_195723](https://huggingface.co/datasets/YDY0427/so101_test_20250708_195723) | apache-2.0 | 50 |
| [YSanYi/so101_push_green_block_into_box](https://huggingface.co/datasets/YSanYi/so101_push_green_block_into_box) | apache-2.0 | 60 |
| [yuk6ra/so101-onetape-cleanup](https://huggingface.co/datasets/yuk6ra/so101-onetape-cleanup) | apache-2.0 | 50 |
| [yuk6ra/so101-pen-cleanup](https://huggingface.co/datasets/yuk6ra/so101-pen-cleanup) | mit | 50 |
| [yuk6ra/so101-tapes-cleanup](https://huggingface.co/datasets/yuk6ra/so101-tapes-cleanup) | apache-2.0 | 50 |
| [yutaro-kimura-acs/so101_pp_blue_and_red](https://huggingface.co/datasets/yutaro-kimura-acs/so101_pp_blue_and_red) | apache-2.0 | 100 |
| [yutaro-kimura-acs/so101_pp_blue_box](https://huggingface.co/datasets/yutaro-kimura-acs/so101_pp_blue_box) | apache-2.0 | 50 |
| [zacapa/SO101_AVE_07](https://huggingface.co/datasets/zacapa/SO101_AVE_07) | apache-2.0 | 226 |
| [zacapa/SO101_chess_test2_6](https://huggingface.co/datasets/zacapa/SO101_chess_test2_6) | apache-2.0 | 274 |
| [zacapa/SO101_chess_test7](https://huggingface.co/datasets/zacapa/SO101_chess_test7) | apache-2.0 | 83 |
| [zacapa/SO101_cube_test2](https://huggingface.co/datasets/zacapa/SO101_cube_test2) | apache-2.0 | 103 |
| [zaringleb/pick_colored_cube_so101](https://huggingface.co/datasets/zaringleb/pick_colored_cube_so101) | apache-2.0 | 60 |
| [zaringleb/pick_single_cube_so101](https://huggingface.co/datasets/zaringleb/pick_single_cube_so101) | apache-2.0 | 185 |
| [ZhangHuTony/so101_pickplace](https://huggingface.co/datasets/ZhangHuTony/so101_pickplace) | apache-2.0 | 50 |
| [zz4321/so101_grasp_rubic](https://huggingface.co/datasets/zz4321/so101_grasp_rubic) | apache-2.0 | 101 |

## Sources recorded by the author

Recorded by Dongyoon Kim on SO-101 hardware and released here under the Apache License 2.0.

| Source | License | Episodes |
|---|---|---|
| pick_place_blue_pen_v1 | apache-2.0 | 18 |
| pickandplace_bluecube_whitecup | apache-2.0 | 11 |
| pickandplace_greencube_whitecup | apache-2.0 | 14 |
| skill_earser_move_v3_followercal | apache-2.0 | 50 |
| skill_eraser_move_v2 | apache-2.0 | 90 |
| stack_cube_normalized | apache-2.0 | 149 |

## Not redistributed

25 further datasets (3,168 episodes) were used to train the model that accompanies this release but are not redistributed here, because their upstream repositories declare no license.
