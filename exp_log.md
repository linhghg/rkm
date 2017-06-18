# SVM
## Gaussian Kernel
`time ./svm-train -c 2 -t 2 -g 0.01 mnist_train_svm_0.dat`
2:33.73

`./svm-predict mnist_test_svm_0.dat mnist_train_svm_0.dat.model output`
99.77%

No shrinking:
`time ./svm-train -c 2 -t 2 -g 0.01 -h 0 mnist_train_svm_0.dat`
4:04.37, 99.77%

## Poly-4 Kernel
`time ./svm-train -c 10 -t 1 -g 1 -r 0 -d 4 mnist_train_svm_0.dat`
3:28.09, 99.69%

`time ./svm-train -c 2 -t 1 -g 1 -r 0 -d 4 mnist_train_svm_0.dat`
3:16.33, 99.69%

# RKM
## No feature selection
fast gaussian, gamma = 1, C = 2, 9:09.04, 0.0481
fast gaussian, gamma = 0.1, C = 2, 11:21.87, 0.0634
fast gaussian, gamma = 10, C = 2, 10:21.87, 0.0327
fast gaussian, gamma = 10, C = 10, 35:50.29, 0.0214
fast gaussian, gamma = 20, C = 2, 12:20.37, 0.0303
fast gaussian, gamma = 10, C = 1, 9:29.50, 0.0399
fast gaussian, gamma = 10, Cp = 10, Cn = 1 22:38.80, 0.0116
fast gaussian, gamma = 10, Cp = 2, Cn = 1 12:56.35, 0.0277
