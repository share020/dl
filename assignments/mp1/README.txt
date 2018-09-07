HW1: Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch).

The neural network should be trained on the Training Set using stochastic gradient descent.
It should achieve 97-98% accuracy on the Test Set.

For full credit, submit via Compass (1) the code and (2) a paragraph (in a PDF document)
which states the Test Accuracy and briefly describes the implementation.

Due September 7 at 5:00 PM.


Result:


$ python main.py
Epoch 1: training cost = 0.265081961482, test cost = 0.263101067507
Epoch 2: training cost = 0.192257499837, test cost = 0.192460479053
Epoch 3: training cost = 0.15334790917, test cost = 0.157792183161
Epoch 4: training cost = 0.129918472063, test cost = 0.137100690363
Epoch 5: training cost = 0.1105753254, test cost = 0.123208760518
Epoch 6: training cost = 0.101754493263, test cost = 0.118015495706
Epoch 7: training cost = 0.0895023220141, test cost = 0.107810720062
Epoch 8: training cost = 0.0802031525878, test cost = 0.0999057023506
Epoch 9: training cost = 0.0713123530575, test cost = 0.0966682614361
Epoch 10: training cost = 0.0707509095556, test cost = 0.0950335272556
Epoch 11: training cost = 0.0618889502614, test cost = 0.0902194327926
Epoch 12: training cost = 0.0594836700699, test cost = 0.0887112213609
Epoch 13: training cost = 0.0533482625207, test cost = 0.0858917731451
Epoch 14: training cost = 0.0497493878107, test cost = 0.0863620849658
Epoch 15: training cost = 0.0456300951972, test cost = 0.0812432592748
Epoch 16: training cost = 0.0433689077762, test cost = 0.0797903665373
Epoch 17: training cost = 0.0399626884096, test cost = 0.0794783895346
Epoch 18: training cost = 0.0384433169915, test cost = 0.0805901787306
Epoch 19: training cost = 0.0350620516271, test cost = 0.0787715005184
Epoch 20: training cost = 0.0325833753773, test cost = 0.0768358601798
Epoch 21: training cost = 0.034084911789, test cost = 0.0809462614759
Epoch 22: training cost = 0.0315315716914, test cost = 0.0805074821726
Epoch 23: training cost = 0.0275477067757, test cost = 0.0745879095451
Epoch 24: training cost = 0.0269855605018, test cost = 0.0789982725824
Epoch 25: training cost = 0.0252056303139, test cost = 0.0752948172436
Epoch 26: training cost = 0.0234739284611, test cost = 0.0744190776858
Epoch 27: training cost = 0.0223856684737, test cost = 0.0757207771773
Epoch 28: training cost = 0.0209431657586, test cost = 0.076040391821
Epoch 29: training cost = 0.0199311334505, test cost = 0.0749167133925
Epoch 30: training cost = 0.0195788886791, test cost = 0.0771335701384
Epoch 31: training cost = 0.0187109933188, test cost = 0.0761018250375
Epoch 32: training cost = 0.0177614238517, test cost = 0.0758681809637
Epoch 33: training cost = 0.0168390727148, test cost = 0.0759259058483
Epoch 34: training cost = 0.0159508795507, test cost = 0.0757620786415
Epoch 35: training cost = 0.0150691695975, test cost = 0.0766713930889
Epoch 36: training cost = 0.0152976682857, test cost = 0.0780594488272
Epoch 37: training cost = 0.0141206563319, test cost = 0.076131049725
Epoch 38: training cost = 0.0132332484026, test cost = 0.0756487528689
Epoch 39: training cost = 0.0129524321047, test cost = 0.0773811524766
Epoch 40: training cost = 0.0122868058048, test cost = 0.0773318415624
Epoch 41: training cost = 0.0124065134115, test cost = 0.0783909511807
Epoch 42: training cost = 0.0117554454087, test cost = 0.0806867460177
Epoch 43: training cost = 0.011807362872, test cost = 0.0775934644233
Epoch 44: training cost = 0.0105890429278, test cost = 0.0780540967234
Epoch 45: training cost = 0.0102032548453, test cost = 0.0781684188151
Epoch 46: training cost = 0.00985393152382, test cost = 0.0782410094465
Epoch 47: training cost = 0.00951154792944, test cost = 0.0800757818157
Epoch 48: training cost = 0.00905277183778, test cost = 0.0781992281772
Epoch 49: training cost = 0.00894101136169, test cost = 0.0790644439324
Epoch 50: training cost = 0.0083409802008, test cost = 0.0792101188431


             precision    recall  f1-score   support

          0       0.99      0.98      0.98       997
          1       0.99      0.99      0.99      1138
          2       0.98      0.98      0.98      1032
          3       0.98      0.97      0.97      1019
          4       0.98      0.98      0.98       984
          5       0.97      0.98      0.97       883
          6       0.98      0.98      0.98       953
          7       0.97      0.98      0.98      1017
          8       0.98      0.98      0.98       974
          9       0.96      0.97      0.97      1003

avg / total       0.98      0.98      0.98     10000