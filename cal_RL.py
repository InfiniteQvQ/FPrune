import numpy as np

a  = np.array([0.5704216361045837, 0.5704216361045837, 0.5704216361045837, 0.5704216361045837, 0.5704216361045837, 0.5704216361045837,
                        0.5704216361045837, 0.6175978779792786, 0.6175978779792786, 0.6175978779792786, 0.6175978779792786, 0.6175978779792786, 
                        0.6175978779792786, 0.6175978779792786, 0.6315311193466187, 0.6315311193466187, 0.6315311193466187, 0.6315311193466187, 
                        0.6315311193466187, 0.6315311193466187, 0.6315311193466187, 0.6307380199432373, 0.6307380199432373, 0.6307380199432373, 
                        0.6307380199432373, 0.6307380199432373, 0.6307380199432373, 0.6307380199432373, 0.6528562903404236, 0.6528562903404236, 
                        0.6528562903404236, 0.6528562903404236, 0.6528562903404236, 0.6528562903404236, 0.6528562903404236, 0.648245096206665, 
                        0.648245096206665, 0.648245096206665, 0.648245096206665, 0.648245096206665, 0.648245096206665, 0.648245096206665, 0.6300591230392456,
                          0.6300591230392456, 0.6300591230392456, 0.6300591230392456, 0.6300591230392456, 0.6300591230392456, 0.6300591230392456, 
                          0.5921671986579895, 0.5921671986579895, 0.5921671986579895, 0.5921671986579895, 0.5921671986579895, 0.5921671986579895, 
                          0.5921671986579895, 0.5973896384239197, 0.5973896384239197, 0.5973896384239197, 0.5973896384239197, 0.5973896384239197, 
                          0.5973896384239197, 0.5973896384239197, 0.5680346488952637, 0.5680346488952637, 0.5680346488952637, 0.5680346488952637, 
                          0.5680346488952637, 0.5680346488952637, 0.5680346488952637, 0.5870822668075562, 0.5870822668075562, 0.5870822668075562, 
                          0.5870822668075562, 0.5870822668075562, 0.5870822668075562, 0.5870822668075562, 0.5893719792366028, 0.5893719792366028, 
                          0.5893719792366028, 0.5893719792366028, 0.5893719792366028, 0.5893719792366028, 0.5893719792366028, 0.5989924073219299, 
                          0.5989924073219299, 0.5989924073219299, 0.5989924073219299, 0.5989924073219299, 0.5989924073219299, 0.5989924073219299, 
                          0.610863208770752, 0.610863208770752, 0.610863208770752, 0.610863208770752, 0.610863208770752, 0.610863208770752, 0.610863208770752, 
                          0.618774950504303, 0.618774950504303, 0.618774950504303, 0.618774950504303, 0.618774950504303, 0.618774950504303, 0.618774950504303, 
                          0.6681280136108398, 0.6681280136108398, 0.6681280136108398, 0.6681280136108398, 0.6681280136108398, 0.6681280136108398, 0.6681280136108398,
                            0.6586800217628479, 0.6586800217628479, 0.6586800217628479, 0.6586800217628479, 0.6586800217628479, 0.6586800217628479, 0.6586800217628479,
                              0.7156056761741638, 0.7156056761741638, 0.7156056761741638, 0.7156056761741638, 0.7156056761741638, 0.7156056761741638, 0.7156056761741638,
                                0.790572464466095, 0.790572464466095, 0.790572464466095, 0.790572464466095, 0.790572464466095, 0.790572464466095, 0.790572464466095, 
                                0.7437890768051147, 0.7437890768051147, 0.7437890768051147, 0.7437890768051147, 0.7437890768051147, 0.7437890768051147, 
                                0.7437890768051147, 0.794614851474762, 0.794614851474762, 0.794614851474762, 0.794614851474762, 0.794614851474762, 
                                0.794614851474762, 0.794614851474762, 0.8248370885848999, 0.8248370885848999, 0.8248370885848999, 0.8248370885848999,
                                  0.8248370885848999, 0.8248370885848999, 0.8248370885848999, 0.7700518369674683, 0.7700518369674683, 0.7700518369674683, 
                                  0.7700518369674683, 0.7700518369674683, 0.7700518369674683, 0.7700518369674683, 0.7629246115684509, 0.7629246115684509, 
                                  0.7629246115684509, 0.7629246115684509, 0.7629246115684509, 0.7629246115684509, 0.7629246115684509, 0.8121660351753235, 
                                  0.8121660351753235, 0.8121660351753235, 0.8121660351753235, 0.8121660351753235, 0.8121660351753235, 0.8121660351753235,
                                    0.8520520329475403, 0.8520520329475403, 0.8520520329475403, 0.8520520329475403, 0.8520520329475403, 0.8520520329475403, 
                                    0.8520520329475403, 0.831261396408081, 0.831261396408081, 0.831261396408081, 0.831261396408081, 0.831261396408081, 
                                    0.831261396408081, 0.831261396408081, 0.8414707183837891, 0.8414707183837891, 0.8414707183837891, 0.8414707183837891,
                                      0.8414707183837891, 0.8414707183837891, 0.8414707183837891, 0.7869208455085754, 0.7869208455085754, 0.7869208455085754,
                                        0.7869208455085754, 0.7869208455085754, 0.7869208455085754, 0.7869208455085754, 0.8296730518341064, 0.8296730518341064, 
                                        0.8296730518341064, 0.8296730518341064, 0.8296730518341064, 0.8296730518341064, 0.8296730518341064, 0.8414230942726135, 
                                        0.8414230942726135, 0.8414230942726135, 0.8414230942726135, 0.8414230942726135, 0.8414230942726135, 0.8414230942726135, 
                                        0.7317030429840088, 0.7317030429840088, 0.7317030429840088, 0.7317030429840088, 0.7317030429840088, 0.7317030429840088, 0.7317030429840088])

b  = []
for i in range(32):
    b.append(a[i*7])

print(np.array(b))