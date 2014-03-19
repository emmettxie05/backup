#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "mkl.h"
#include "pttSymSGD.h"

#define PAGESIZE 4096
int panels, T, nt, panelSz;
float prev_err = 0.0;

// X: a MxD matrix, Y: a M vector, W: a M vector
// W0: a M vector
int main(int argc, char ** argv){
    if (argc>1 && argv[1][0]=='h') {
        printf ("Usage: parSymSGD M D T C lamda r\n");
        printf ("  M: number of data points, D: dimensions, T: time iterations, C: cores;\n");
        printf ("  lamda: learning rate, r: panel size in unit of C.\n");
        return 1;
    }u
    // read in the arguments: M, D, I (time iterations), C (cores), r (each panel contains r*C points)
    int M = argc>1?atoi(argv[1]):32;
    int D = argc>2?atoi(argv[2]):4;
    T = argc>3?atoi(argv[3]):10;
    int C = argc>4?atoi(argv[4]):4;
    float lamda = argc>5?atof(argv[5]):0.01;
    int r = argc>6?atoi(argv[6]):1;
    ///printf("M=%d, D=%d, T=%d, C=%d, lamda=%8.6f, r=%d\n",M,D,T,C,lamda,r);

    int max_threads = mkl_get_max_threads(); // get the max number of threads
	
    int rep;
    mkl_set_num_threads(1); // set the number of threads to use by mkl
    panelSz = C*r;
    panels = M/panelSz;

    int i,j,k,p,t;
    float *Y, *Wreal, *W, *X;
    Y = (float *) mkl_malloc(M*sizeof(float),PAGESIZE);
    Wreal = (float *) mkl_malloc(D*sizeof(float),PAGESIZE);
    W = (float *) mkl_malloc(D*sizeof(float),PAGESIZE);
    X = (float *) mkl_malloc(M*D*sizeof(float),PAGESIZE);
    float *Ypred = (float*)mkl_malloc(M*sizeof(float),PAGESIZE);
    float *Ytmp = (float*)mkl_malloc(M*sizeof(float),PAGESIZE);
	float *I = (float*)mkl_malloc(D*D*sizeof(float),PAGESIZE);
    float *Z = (float*)mkl_malloc(M*D*sizeof(float),PAGESIZE);
    float *B = (float*)mkl_malloc(panels*D*sizeof(float),PAGESIZE);

    if (Y==NULL | Wreal==NULL | W==NULL | X==NULL | Ypred==NULL || Ytmp==NULL || Z==NULL || B==NULL || I== NULL){
        printf("Memory allocation error.\n");
        return 2;
    }

    initData(Wreal,W,X,Y, M, D,I);

    ///printf("panelSz=%d, panels=%d\n", panelSz, panels);

    for (nt=1; nt<=max_threads && nt<=panelSz; nt*=2){
        omp_set_num_threads(nt);// set the number of openMP threads

        for (rep=0; rep<REPEATS; rep++){//repeat measurements
            double prepTime, gdTime, sInit;
            // preprocessing
            sInit=dsecnd();
            //preprocessSeq(X, Y, Z, B, panelSz, panels, M, D, lamda);
            preprocessPar(X, Y, Z, B, panelSz, panels, M, D, lamda);
            prepTime = (dsecnd() - sInit);
            ///dump2("Z",Z,M,D);
            ///dump2("B",B,panels,D);

            // GD
            initW(W,D);
            ///dump1("W (initial)", W, D);
            sInit=dsecnd();
            float err;
            float fixpoint = 0.0;
            for (t=0;t<T;t++){
                for (p=0;p<panels;p++){
                    gd(&(X[p*panelSz*D]),&(Z[p*panelSz*D]), &(B[p*D]), panelSz, D, lamda, W, I);
                    ///printf("(t=%d, p=%d) ",t,p);
                    ///dump1("W", W, D);
                    ///err=calErr(X, Ypred, Ytmp, Y, W, M, D);
                  printf("finish  one  panels     ............................  \n");
                }
            }
            gdTime = (dsecnd() - sInit);

            err=calErr(X, Ypred, Ytmp, Y, W, M, D);
            fixpoint = err - prev_err;
            

            // print final err. time is in milliseconds
            printf("nt=%d\t ttlTime=%.5f\t prepTime=%.5f\t gdTime=%.5f\t error=%.5f\n", nt, (gdTime+prepTime)*1000, prepTime*1000, gdTime*1000, err);
        }
    }
    if (B) mkl_free(B);
    if (Z) mkl_free(Z);
    if (Ytmp) mkl_free(Ytmp);
    if (Ypred) mkl_free(Ypred);
    if (Y) mkl_free(Y);
    if (Wreal) mkl_free(Wreal);
    if (W) mkl_free(W);
    if (X) mkl_free(X);
	if (I) mkl_free(I);
    return 0;
}

void initData(float*Wreal, float*W, float*X, float*Y, int M, int D, float*I){
    int i;
    srand(1);
    float u1=4;// upper bound of X element values
    float u2=0.1; // upper bound of noise in Y
    for (i=0;i<D;i++){
        Wreal[i] = (float)floor(u1*rand()/RAND_MAX-u1/2);
    }
    for (i=0;i<M;i++){
        Y[i] = u2*rand()/RAND_MAX;
    }
    for (i=0;i<M*D;i++){
        X[i] = u1*rand()/RAND_MAX;
    }
    float alpha = 1.0;
    float beta = 0;
    ///dump1("Wreal", Wreal, D);
    ///dump1("Y (initial)", Y, M);
    ///dump2("X", X, M,D);
	// Y= X*Wreal + Y
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, D, alpha, X, D,  Wreal, 1, 1, Y, 1);
    ///dump1("Y (real)", Y, M);
     //printf("finish the initial \n");
    }

void preprocessPar(float *X, float*Y, float *Z, float *B, int panelSz, int panels, int M, int D, float lamda){
    int p,i;
    for (i=0;i<panels*D;i++)
        B[i] = 0;
    // this line will need some changes to be more space efficient (e.g., applying the paralleliszation to inside a panel rather than across panels.) 
    //printf("nt=%d\n", nt);
    float *XX = (float*) malloc(nt*panelSz*panelSz*sizeof(float)); //holder of X[j]X[j] for one panel
    if (XX==NULL){
        printf("Memory allocation error.\n");
        return 5;
    }
    int chunkSz=panels/nt;
#pragma omp parallel for private(i,p)
    for (i=0;i<nt;i++){
        for (p=i*chunkSz;p<(i+1)*chunkSz;p++){
            ///printf("preprocess panel %d...\n", p);
            ///printf("%f\n", X[p*panelSz*D]);
            preprocess1(panelSz, D, &(XX[i*panelSz*panelSz]), &(X[p*panelSz*D]), &(Y[p*panelSz]), lamda, &(Z[p*panelSz*D]), &(B[p*D]));
        }
    }
    if (XX) free(XX);
}

void preprocess1(int panelSz, int D, float *XX, float*X, float*Y, float lamda, float*Z, float*B){
    // panelSz: the number of points to process in each round
    int i,j;
    // step 1: compute all X[i]'*X[j] (i>j)
    for (i=panelSz-1;i>0;i--)
        for (j=i-1;j>=0;j--){
            XX[i*panelSz+j] = cblas_sdot(D, &(X[i*D]), 1, &(X[j*D]), 1);
         // printf("XX[%d]=%8.4f, X[%d]=%8.4f, X[%d]=%8.4f\n", i*panelSz+j, XX[i*panelSz+j], i*D, X[i*D], j*D, X[j*D]);
        }

	
    // step 2: compute all Z vectors
    // Z0=lamda*X[0], B=lamda*X[0]*Y[0]
    cblas_scopy(D, X, 1, Z, 1);  

    cblas_sscal(D, lamda, Z, 1);
    float alpha=lamda*Y[0];
    cblas_scopy(D, X, 1, B, 1);
    cblas_sscal(D, alpha, B, 1);
    for (i=1; i<panelSz;i++){
        cblas_scopy(D, &(X[i*D]), 1, &(Z[i*D]),1);
        // Z[i] = lamda*(X[i] - sum_{j<i} XX[i,j]*Z[j]);
        for (j=0;j<i;j++){
            cblas_saxpy(D, -1*XX[i*panelSz+j], &(Z[j*D]), 1, &(Z[i*D]), 1);
        }
        cblas_sscal(D, lamda, &(Z[i*D]), 1);
        // B = lamda*(Y[i] - X[i]*B) X[i] + B;
        float alpha = lamda*(Y[i]-cblas_sdot(D, &(X[i*D]), 1, B, 1));
        cblas_saxpy(D, alpha, &(X[i*D]), 1, B, 1);
    }
}

void gd(float *X, float* Z, float*B, int panelSz, int D, float lamda, float * W, float *I){
    // each D elements in Z forms a z vector
	// W=(I-sum_{j}Z[j]X[j])W[0]+B
    int i,j,k,m,n,l;
    float temp;
    float *Wtmp = (float*)malloc(D*sizeof(float));
    
    memset(Wtmp,NULL,D*sizeof(float));
    // every iteration should re-initial I
    for(i=0;i<D*D;i++){
      I[i]= 0.0;
      //I[i*D+i]= 1.0;
      }

      for(i=0;i<D;i++){
        I[i*D+i]= 1.0;
        }

 // for(i=0;i<D;i++){
 //    printf("W[%d]= %8.4f \n", i, W[i]);
 //  }



   int chunkSz= D/nt;
   float *XZ = (float*) malloc(D*D*sizeof(float)); //holder of matrix XZ
   memset(XZ,NULL,D*D*sizeof(float));


int PchunkSz = panelSz/nt;
#pragma omp parallel for schedule(static) private(j,i)
 for(k=0;k<nt;k++){
 for(i=0;i<D;i++){
  for(m=0;m<D;m++){
    temp = 0;
    for(j=k*PchunkSz;j<(k+1)*PchunkSz;j++){
     temp += X[i*panelSz+j]*Z[j*D+m];
     }
     XZ[i*D+m] = temp;
     //printf("XZ[%i*D+%j] = %8.4f \n",i,j,XZ[i*D+j]);
  }
 }
 }


 #pragma omp parallel for schedule(static) private(i)
  for(k=0;k<nt;k++){
    for(i=k*chunkSz;i<(k+1)*chunkSz;i++){
       // printf("first,I[%d]=%8.4f \n", i, I[i]);
       for(j=0;j<chunkSz*D;j++)
          I[i*D+j]+= -XZ[i*D+j];

    }
  }  
 
    cblas_saxpy(D,1,B, 1, Wtmp, 1);
    cblas_scopy(D, Wtmp, 1, W, 1);
   
     for(i=0;i<D;i++){
     //printf("Wtmp[%d]=%8.4f \n",i,Wtmp[i]);
    }
    free(XZ);
    free(Wtmp);

   

}



float calErr(float *data, float *Ypred, float *Ytmp, float* Y, float* W, int M, int D){
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, D, 1, data, D, W, 1, 0, Ypred, 1);
    cblas_scopy(M,Ypred,1,Ytmp,1);
    cblas_saxpy(M,-1,Y,1,Ytmp,1);
    return cblas_sdot(M,Ytmp,1,Ytmp,1);
}

void initW(float*W, int D){
    // initialize W
    int i;
    srand(2);
    float u1=4;
    for (i=0;i<D;i++){
        W[i] = (float)floor(u1*rand()/RAND_MAX-u1/2);
    }
}

void dump1(char*s, float*V, int D){
    printf("%s:\n", s);
    int i;
    for (i=0;i<D;i++)
        printf("%8.4f  ", V[i]);
    printf("\n-------------\n");
}

void dump2(char*s, float*A, int M, int D){
    printf("%s:\n", s);
    int i,j;
    for (i=0;i<M;i++){
        printf("%d:\t", i);
        for (j=0;j<D;j++){
            float f = A[i*D+j];
            printf("%8.4f  ", f);
        }
        printf("\n");
    }
    printf("--------------\n");
}

void preprocessSeq(float *X, float*Y, float *Z, float *B, int panelSz, int panels, int M, int D, float lamda){
    int p,i;
    for (i=0;i<panels*D;i++)
        B[i] = 0;
    float *XX = (float*) malloc(panelSz*panelSz*sizeof(float)); //holder of X[j]X[j] for one panel
    if (XX==NULL){
        printf("Memory allocation error.\n");
        return 5;
    }
    for (p=0;p<panels;p++){
        ///printf("preprocess panel %d...\n", p);
        ///printf("%f\n", X[p*panelSz]);
        preprocess1(panelSz, D, XX, &(X[p*panelSz*D]), &(Y[p*panelSz]), lamda, &(Z[p*panelSz*D]), &(B[p*D]));
    }
    if (XX) free(XX);
}
