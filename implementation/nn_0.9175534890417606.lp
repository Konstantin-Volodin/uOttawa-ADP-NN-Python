\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  20 sy[p1] + 10 sy[p2] + 5 sy[p3] - 20 aa[p1,0] - 20 aa[p1,1] - 20 aa[p1,2]
   - 20 aa[p1,3] - 20 aa[p1,4] - 20 aa[p1,5] - 20 aa[p1,6] + 19.8 aa[p1,8]
   + 39.402 aa[p1,9] + 58.80798 aa[p1,10] + 78.0199002 aa[p1,11]
   + 97.03970119799999 aa[p1,12] + 115.86930418602 aa[p1,13]
   + 134.5106111441598 aa[p1,14] + 152.9655050327182 aa[p1,15]
   + 171.235849982391 aa[p1,16] + 189.3234914825671 aa[p1,17]
   + 207.2302565677414 aa[p1,18] + 224.957954002064 aa[p1,19]
   + 242.5083744620433 aa[p1,20] + 259.8832907174229 aa[p1,21]
   + 277.0844578102486 aa[p1,22] + 294.1136132321461 aa[p1,23]
   + 310.9724770998247 aa[p1,24] + 327.6627523288265 aa[p1,25]
   + 344.1861248055382 aa[p1,26] + 360.5442635574829 aa[p1,27]
   + 376.738820921908 aa[p1,28] + 392.7714327126889 aa[p1,29] - 10 aa[p2,0]
   - 10 aa[p2,1] - 10 aa[p2,2] - 10 aa[p2,3] - 10 aa[p2,4] - 10 aa[p2,5]
   - 10 aa[p2,6] - 10 aa[p2,7] - 10 aa[p2,8] - 10 aa[p2,9] - 10 aa[p2,10]
   - 10 aa[p2,11] - 10 aa[p2,12] - 10 aa[p2,13] + 9.9 aa[p2,15]
   + 19.701 aa[p2,16] + 29.40399 aa[p2,17] + 39.0099501 aa[p2,18]
   + 48.51985059899999 aa[p2,19] + 57.93465209300999 aa[p2,20]
   + 67.25530557207989 aa[p2,21] + 76.48275251635908 aa[p2,22]
   + 85.61792499119549 aa[p2,23] + 94.66174574128354 aa[p2,24]
   + 103.6151282838707 aa[p2,25] + 112.478977001032 aa[p2,26]
   + 121.2541872310217 aa[p2,27] + 129.9416453587114 aa[p2,28]
   + 138.5422289051243 aa[p2,29] - 5 aa[p3,0] - 5 aa[p3,1] - 5 aa[p3,2]
   - 5 aa[p3,3] - 5 aa[p3,4] - 5 aa[p3,5] - 5 aa[p3,6] - 5 aa[p3,7]
   - 5 aa[p3,8] - 5 aa[p3,9] - 5 aa[p3,10] - 5 aa[p3,11] - 5 aa[p3,12]
   - 5 aa[p3,13] - 5 aa[p3,14] - 5 aa[p3,15] - 5 aa[p3,16] - 5 aa[p3,17]
   - 5 aa[p3,18] - 5 aa[p3,19] - 5 aa[p3,20] + 4.95 aa[p3,22]
   + 9.8505 aa[p3,23] + 14.701995 aa[p3,24] + 19.50497505 aa[p3,25]
   + 24.2599252995 aa[p3,26] + 28.96732604650499 aa[p3,27]
   + 33.62765278603995 aa[p3,28] + 38.24137625817954 aa[p3,29] + 80 az[p1]
   + 90 az[p2] + 95 az[p3] + 0.99 cost-to-go
Subject To
 c_cap[0]: sx[0] + aa[p1,0] + aa[p2,0] + aa[p3,0] <= 10
 c_cap[1]: sx[1] + aa[p1,1] + aa[p2,1] + aa[p3,1] <= 10
 c_cap[2]: sx[2] + aa[p1,2] + aa[p2,2] + aa[p3,2] <= 10
 c_cap[3]: sx[3] + aa[p1,3] + aa[p2,3] + aa[p3,3] <= 10
 c_cap[4]: sx[4] + aa[p1,4] + aa[p2,4] + aa[p3,4] <= 10
 c_cap[5]: sx[5] + aa[p1,5] + aa[p2,5] + aa[p3,5] <= 10
 c_cap[6]: sx[6] + aa[p1,6] + aa[p2,6] + aa[p3,6] <= 10
 c_cap[7]: sx[7] + aa[p1,7] + aa[p2,7] + aa[p3,7] <= 10
 c_cap[8]: sx[8] + aa[p1,8] + aa[p2,8] + aa[p3,8] <= 10
 c_cap[9]: sx[9] + aa[p1,9] + aa[p2,9] + aa[p3,9] <= 10
 c_cap[10]: sx[10] + aa[p1,10] + aa[p2,10] + aa[p3,10] <= 10
 c_cap[11]: sx[11] + aa[p1,11] + aa[p2,11] + aa[p3,11] <= 10
 c_cap[12]: sx[12] + aa[p1,12] + aa[p2,12] + aa[p3,12] <= 10
 c_cap[13]: sx[13] + aa[p1,13] + aa[p2,13] + aa[p3,13] <= 10
 c_cap[14]: sx[14] + aa[p1,14] + aa[p2,14] + aa[p3,14] <= 10
 c_cap[15]: sx[15] + aa[p1,15] + aa[p2,15] + aa[p3,15] <= 10
 c_cap[16]: sx[16] + aa[p1,16] + aa[p2,16] + aa[p3,16] <= 10
 c_cap[17]: sx[17] + aa[p1,17] + aa[p2,17] + aa[p3,17] <= 10
 c_cap[18]: sx[18] + aa[p1,18] + aa[p2,18] + aa[p3,18] <= 10
 c_cap[19]: sx[19] + aa[p1,19] + aa[p2,19] + aa[p3,19] <= 10
 c_cap[20]: sx[20] + aa[p1,20] + aa[p2,20] + aa[p3,20] <= 10
 c_cap[21]: sx[21] + aa[p1,21] + aa[p2,21] + aa[p3,21] <= 10
 c_cap[22]: sx[22] + aa[p1,22] + aa[p2,22] + aa[p3,22] <= 10
 c_cap[23]: sx[23] + aa[p1,23] + aa[p2,23] + aa[p3,23] <= 10
 c_cap[24]: sx[24] + aa[p1,24] + aa[p2,24] + aa[p3,24] <= 10
 c_cap[25]: sx[25] + aa[p1,25] + aa[p2,25] + aa[p3,25] <= 10
 c_cap[26]: sx[26] + aa[p1,26] + aa[p2,26] + aa[p3,26] <= 10
 c_cap[27]: sx[27] + aa[p1,27] + aa[p2,27] + aa[p3,27] <= 10
 c_cap[28]: sx[28] + aa[p1,28] + aa[p2,28] + aa[p3,28] <= 10
 c_cap[29]: sx[29] + aa[p1,29] + aa[p2,29] + aa[p3,29] <= 10
 c_scap: az[p1] + az[p2] + az[p3] <= 15
 c_dem[p1]: - sy[p1] + aa[p1,0] + aa[p1,1] + aa[p1,2] + aa[p1,3] + aa[p1,4]
   + aa[p1,5] + aa[p1,6] + aa[p1,7] + aa[p1,8] + aa[p1,9] + aa[p1,10]
   + aa[p1,11] + aa[p1,12] + aa[p1,13] + aa[p1,14] + aa[p1,15] + aa[p1,16]
   + aa[p1,17] + aa[p1,18] + aa[p1,19] + aa[p1,20] + aa[p1,21] + aa[p1,22]
   + aa[p1,23] + aa[p1,24] + aa[p1,25] + aa[p1,26] + aa[p1,27] + aa[p1,28]
   + aa[p1,29] + az[p1] <= 0
 c_dem[p2]: - sy[p2] + aa[p2,0] + aa[p2,1] + aa[p2,2] + aa[p2,3] + aa[p2,4]
   + aa[p2,5] + aa[p2,6] + aa[p2,7] + aa[p2,8] + aa[p2,9] + aa[p2,10]
   + aa[p2,11] + aa[p2,12] + aa[p2,13] + aa[p2,14] + aa[p2,15] + aa[p2,16]
   + aa[p2,17] + aa[p2,18] + aa[p2,19] + aa[p2,20] + aa[p2,21] + aa[p2,22]
   + aa[p2,23] + aa[p2,24] + aa[p2,25] + aa[p2,26] + aa[p2,27] + aa[p2,28]
   + aa[p2,29] + az[p2] <= 0
 c_dem[p3]: - sy[p3] + aa[p3,0] + aa[p3,1] + aa[p3,2] + aa[p3,3] + aa[p3,4]
   + aa[p3,5] + aa[p3,6] + aa[p3,7] + aa[p3,8] + aa[p3,9] + aa[p3,10]
   + aa[p3,11] + aa[p3,12] + aa[p3,13] + aa[p3,14] + aa[p3,15] + aa[p3,16]
   + aa[p3,17] + aa[p3,18] + aa[p3,19] + aa[p3,20] + aa[p3,21] + aa[p3,22]
   + aa[p3,23] + aa[p3,24] + aa[p3,25] + aa[p3,26] + aa[p3,27] + aa[p3,28]
   + aa[p3,29] + az[p3] <= 0
 c_sxp[0]: - sx[0] - aa[p1,0] - aa[p2,0] - aa[p3,0] + sxp[0] = 0
 c_sxp[1]: - sx[1] - aa[p1,1] - aa[p2,1] - aa[p3,1] + sxp[1] = 0
 c_sxp[2]: - sx[2] - aa[p1,2] - aa[p2,2] - aa[p3,2] + sxp[2] = 0
 c_sxp[3]: - sx[3] - aa[p1,3] - aa[p2,3] - aa[p3,3] + sxp[3] = 0
 c_sxp[4]: - sx[4] - aa[p1,4] - aa[p2,4] - aa[p3,4] + sxp[4] = 0
 c_sxp[5]: - sx[5] - aa[p1,5] - aa[p2,5] - aa[p3,5] + sxp[5] = 0
 c_sxp[6]: - sx[6] - aa[p1,6] - aa[p2,6] - aa[p3,6] + sxp[6] = 0
 c_sxp[7]: - sx[7] - aa[p1,7] - aa[p2,7] - aa[p3,7] + sxp[7] = 0
 c_sxp[8]: - sx[8] - aa[p1,8] - aa[p2,8] - aa[p3,8] + sxp[8] = 0
 c_sxp[9]: - sx[9] - aa[p1,9] - aa[p2,9] - aa[p3,9] + sxp[9] = 0
 c_sxp[10]: - sx[10] - aa[p1,10] - aa[p2,10] - aa[p3,10] + sxp[10] = 0
 c_sxp[11]: - sx[11] - aa[p1,11] - aa[p2,11] - aa[p3,11] + sxp[11] = 0
 c_sxp[12]: - sx[12] - aa[p1,12] - aa[p2,12] - aa[p3,12] + sxp[12] = 0
 c_sxp[13]: - sx[13] - aa[p1,13] - aa[p2,13] - aa[p3,13] + sxp[13] = 0
 c_sxp[14]: - sx[14] - aa[p1,14] - aa[p2,14] - aa[p3,14] + sxp[14] = 0
 c_sxp[15]: - sx[15] - aa[p1,15] - aa[p2,15] - aa[p3,15] + sxp[15] = 0
 c_sxp[16]: - sx[16] - aa[p1,16] - aa[p2,16] - aa[p3,16] + sxp[16] = 0
 c_sxp[17]: - sx[17] - aa[p1,17] - aa[p2,17] - aa[p3,17] + sxp[17] = 0
 c_sxp[18]: - sx[18] - aa[p1,18] - aa[p2,18] - aa[p3,18] + sxp[18] = 0
 c_sxp[19]: - sx[19] - aa[p1,19] - aa[p2,19] - aa[p3,19] + sxp[19] = 0
 c_sxp[20]: - sx[20] - aa[p1,20] - aa[p2,20] - aa[p3,20] + sxp[20] = 0
 c_sxp[21]: - sx[21] - aa[p1,21] - aa[p2,21] - aa[p3,21] + sxp[21] = 0
 c_sxp[22]: - sx[22] - aa[p1,22] - aa[p2,22] - aa[p3,22] + sxp[22] = 0
 c_sxp[23]: - sx[23] - aa[p1,23] - aa[p2,23] - aa[p3,23] + sxp[23] = 0
 c_sxp[24]: - sx[24] - aa[p1,24] - aa[p2,24] - aa[p3,24] + sxp[24] = 0
 c_sxp[25]: - sx[25] - aa[p1,25] - aa[p2,25] - aa[p3,25] + sxp[25] = 0
 c_sxp[26]: - sx[26] - aa[p1,26] - aa[p2,26] - aa[p3,26] + sxp[26] = 0
 c_sxp[27]: - sx[27] - aa[p1,27] - aa[p2,27] - aa[p3,27] + sxp[27] = 0
 c_sxp[28]: - sx[28] - aa[p1,28] - aa[p2,28] - aa[p3,28] + sxp[28] = 0
 c_sxp[29]: - sx[29] - aa[p1,29] - aa[p2,29] - aa[p3,29] + sxp[29] = 0
 c_syp[p1]: - sy[p1] + aa[p1,0] + aa[p1,1] + aa[p1,2] + aa[p1,3] + aa[p1,4]
   + aa[p1,5] + aa[p1,6] + aa[p1,7] + aa[p1,8] + aa[p1,9] + aa[p1,10]
   + aa[p1,11] + aa[p1,12] + aa[p1,13] + aa[p1,14] + aa[p1,15] + aa[p1,16]
   + aa[p1,17] + aa[p1,18] + aa[p1,19] + aa[p1,20] + aa[p1,21] + aa[p1,22]
   + aa[p1,23] + aa[p1,24] + aa[p1,25] + aa[p1,26] + aa[p1,27] + aa[p1,28]
   + aa[p1,29] + az[p1] + syp[p1] = 0
 c_syp[p2]: - sy[p2] + aa[p2,0] + aa[p2,1] + aa[p2,2] + aa[p2,3] + aa[p2,4]
   + aa[p2,5] + aa[p2,6] + aa[p2,7] + aa[p2,8] + aa[p2,9] + aa[p2,10]
   + aa[p2,11] + aa[p2,12] + aa[p2,13] + aa[p2,14] + aa[p2,15] + aa[p2,16]
   + aa[p2,17] + aa[p2,18] + aa[p2,19] + aa[p2,20] + aa[p2,21] + aa[p2,22]
   + aa[p2,23] + aa[p2,24] + aa[p2,25] + aa[p2,26] + aa[p2,27] + aa[p2,28]
   + aa[p2,29] + az[p2] + syp[p2] = 0
 c_syp[p3]: - sy[p3] + aa[p3,0] + aa[p3,1] + aa[p3,2] + aa[p3,3] + aa[p3,4]
   + aa[p3,5] + aa[p3,6] + aa[p3,7] + aa[p3,8] + aa[p3,9] + aa[p3,10]
   + aa[p3,11] + aa[p3,12] + aa[p3,13] + aa[p3,14] + aa[p3,15] + aa[p3,16]
   + aa[p3,17] + aa[p3,18] + aa[p3,19] + aa[p3,20] + aa[p3,21] + aa[p3,22]
   + aa[p3,23] + aa[p3,24] + aa[p3,25] + aa[p3,26] + aa[p3,27] + aa[p3,28]
   + aa[p3,29] + az[p3] + syp[p3] = 0
 c_sxp_t: - sxp[0] - sxp[1] - sxp[2] - sxp[3] - sxp[4] - sxp[5] - sxp[6]
   - sxp[7] - sxp[8] - sxp[9] - sxp[10] - sxp[11] - sxp[12] - sxp[13]
   - sxp[14] - sxp[15] - sxp[16] - sxp[17] - sxp[18] - sxp[19] - sxp[20]
   - sxp[21] - sxp[22] - sxp[23] - sxp[24] - sxp[25] - sxp[26] - sxp[27]
   - sxp[28] - sxp[29] + sxp_t = 0
 c_syp_t: - syp[p1] - syp[p2] - syp[p3] + syp_t = 0
 def-cost-to-go: 3.5440408227112725e+15 sxp[0]
   + 3.54404082271126e+15 sxp[1] + 3.5440408227112575e+15 sxp[2]
   + 3.544040822711255e+15 sxp[3] + 3.5440408227112455e+15 sxp[4]
   - 46725.60144042969 sxp[5] - 1.786217145675016e+15 syp[p1]
   - 1.7862171456750165e+15 syp[p2] - 1.7862171456750193e+15 syp[p3]
   - 3.544040822711268e+15 sxp_t + 1.7862171456749845e+15 syp_t
   + cost-to-go = 0
Bounds
 sx[0] = 9
 sx[1] = 10
 sx[2] = 10
 sx[3] = 10
 sx[4] = 6
 sx[5] = 8
 sx[6] = 7
 sx[7] = 3
 sx[8] = 5
 sx[9] = 3
 sx[10] = 3
 sx[11] = 6
 sx[12] = 2
 sx[13] = 7
 sx[14] = 3
 sx[15] = 3
 sx[16] = 7
 sx[17] = 4
 sx[18] = 5
 sx[19] = 3
 sx[20] = 3
 sx[21] = 6
 sx[22] = 2
 sx[23] = 1
 sx[24] = 6
 sx[25] = 2
 sx[26] = 0
 sx[27] = 0
 sx[28] = 0
 sx[29] = 0
 sy[p1] = 0
 sy[p2] = 4
 sy[p3] = 3
 cost-to-go free
Generals
 sx[0] sx[1] sx[2] sx[3] sx[4] sx[5] sx[6] sx[7] sx[8] sx[9] sx[10] sx[11]
 sx[12] sx[13] sx[14] sx[15] sx[16] sx[17] sx[18] sx[19] sx[20] sx[21]
 sx[22] sx[23] sx[24] sx[25] sx[26] sx[27] sx[28] sx[29] sy[p1] sy[p2]
 sy[p3] aa[p1,0] aa[p1,1] aa[p1,2] aa[p1,3] aa[p1,4] aa[p1,5] aa[p1,6]
 aa[p1,7] aa[p1,8] aa[p1,9] aa[p1,10] aa[p1,11] aa[p1,12] aa[p1,13]
 aa[p1,14] aa[p1,15] aa[p1,16] aa[p1,17] aa[p1,18] aa[p1,19] aa[p1,20]
 aa[p1,21] aa[p1,22] aa[p1,23] aa[p1,24] aa[p1,25] aa[p1,26] aa[p1,27]
 aa[p1,28] aa[p1,29] aa[p2,0] aa[p2,1] aa[p2,2] aa[p2,3] aa[p2,4] aa[p2,5]
 aa[p2,6] aa[p2,7] aa[p2,8] aa[p2,9] aa[p2,10] aa[p2,11] aa[p2,12]
 aa[p2,13] aa[p2,14] aa[p2,15] aa[p2,16] aa[p2,17] aa[p2,18] aa[p2,19]
 aa[p2,20] aa[p2,21] aa[p2,22] aa[p2,23] aa[p2,24] aa[p2,25] aa[p2,26]
 aa[p2,27] aa[p2,28] aa[p2,29] aa[p3,0] aa[p3,1] aa[p3,2] aa[p3,3] aa[p3,4]
 aa[p3,5] aa[p3,6] aa[p3,7] aa[p3,8] aa[p3,9] aa[p3,10] aa[p3,11] aa[p3,12]
 aa[p3,13] aa[p3,14] aa[p3,15] aa[p3,16] aa[p3,17] aa[p3,18] aa[p3,19]
 aa[p3,20] aa[p3,21] aa[p3,22] aa[p3,23] aa[p3,24] aa[p3,25] aa[p3,26]
 aa[p3,27] aa[p3,28] aa[p3,29] az[p1] az[p2] az[p3] sxp[0] sxp[1] sxp[2]
 sxp[3] sxp[4] sxp[5] sxp[6] sxp[7] sxp[8] sxp[9] sxp[10] sxp[11] sxp[12]
 sxp[13] sxp[14] sxp[15] sxp[16] sxp[17] sxp[18] sxp[19] sxp[20] sxp[21]
 sxp[22] sxp[23] sxp[24] sxp[25] sxp[26] sxp[27] sxp[28] sxp[29] syp[p1]
 syp[p2] syp[p3] sxp_t syp_t
End
