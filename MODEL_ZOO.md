# M3DETR Model Zoo and Baselines

## Introduction

This file documents a collection of models reported in our paper.

#### How to Read the Tables
* The "Name" column contains a link to the config file. 
*  Note that ”l” and ”h” represent layer number and head dimension of M3 Transformers, respectively. "Top" denotes the number of proposals used for keypoint sampling from RPN stage. ”Rel. Trans.” and ”Rep. and Scal. Trans.” refer to mutual-relation trans-former, and multi-representation and multi- scale transformer, respectively. "mAP" represents 3D mean average precision (mAP) on LEVEL_1 difficulty in the Vehicle class with IoU threshold of 0.7 on the full 202 Waymo Validation Set.

## Waymo Open Dataset Model Zoo

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">L_Rel_Trans</th>
<th valign="bottom">H_Rel_Trans</th>
<th valign="bottom">L_Scal_Trans</th>
<th valign="bottom">H_Scal_Trans</th>
<th valign="bottom">Top</th>
<th valign="bottom">mAP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_R50_bs16_50ep -->
 <tr><td align="left"><a href="https://github.com/rayguan97/M3DETR/blob/main/tools/cfgs/m3detr_models/m3detr_waymo_1000.yaml">M3DETR</a></td>
<td align="center">2</td>
<td align="center">4</td>
<td align="center">2</td>
<td align="center">4</td>
<td align="center">1000</td>
<td align="center">75.68</td>
<td align="center"><a href="https://drive.google.com/file/d/12APHpdXdxHRBmchg5xtBEDn9gOUtFwm3/view?usp=sharing">model</a></td>
</tr>

<!-- TABLE BODY -->
<!-- ROW: maskformer2_R50_bs16_50ep -->
 <tr><td align="left"><a href="https://github.com/rayguan97/M3DETR/blob/main/tools/cfgs/m3detr_models/m3detr_waymo_1500.yaml">M3DETR</a></td>
<td align="center">2</td>
<td align="center">4</td>
<td align="center">2</td>
<td align="center">4</td>
<td align="center">1500</td>
<td align="center">75.71</td>
<td align="center"><a href="https://drive.google.com/file/d/1jxAYI6tdplXD6nCmyvjyq8Djp514i8bP/view?usp=sharing">model</a></td>
</tr>

</tbody></table>
