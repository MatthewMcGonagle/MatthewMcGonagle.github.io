example.exe : example.cu jacobi.obj
	nvcc example.cu jacobi.obj -o example.exe

jacobi.obj : jacobi.cu jacobi.cuh
	nvcc -dc jacobi.cu

.PHONY: cleangraphs
cleangraphs:
	rm graphs/*.svg

