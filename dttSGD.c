#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "mpi.h"
#include "mkl.h"
#include "dttSymSGD.h"

#define PAGESIZE 4096
#define DIETAG 2
#define PREPARETAG 3
#define X_TAG 5
#define W_TAG 6
#define Y_TAG 7
#define G_TAG 8
#define Z_TAG 9
#define B_TAG 10
#define XZ_TAG 11

int process_num, total_process;

double prepTime, gdTime, sInit;
//number of nodes.
int M, D, T, C, r, panels, panelSz, PchunkSz, chunkPreSz;
float prev_err = 0.0;
float lamda;

void master() {
    int i,j,k,p,t;
    int rep = 5;
    float *Y, *Wreal, *W, *X, err,*I;

    MPI_Status status;
    printf("master begin \n");
    Y = (float *) malloc(M*sizeof(float));
    Wreal = (float *) malloc(D*sizeof(float));
    W = (float *) malloc(D*sizeof(float));
    X = (float *) malloc(M*D*sizeof(float));
   // I = (float *) malloc(D*D*sizeof(float));
    float *B = (float*)malloc(panels*D*sizeof(float));
    float *Ytmp = (float*)malloc(M*sizeof(float));
    float *Z = (float*)malloc(M*D*sizeof(float));
    float *Ypred = (float*)malloc(M*sizeof(float));

    if (Y==NULL | Wreal==NULL | W==NULL | X==NULL | Ypred==NULL || Ytmp==NULL 
        || Z==NULL || B==NULL){
        printf("Memory allocation error.\n");
        return 2;
    }
  
    initData(Wreal,W,X,Y, M, D);

    float *W_tmp = (float*) malloc(D*sizeof(float)); //holder of X[j]X[j] for one panel
    
        double prepTime, gdTime, sInit;
        // preprocessing
start:
        sInit=dsecnd();
        preprocessPar(X, Y, Z, B, panelSz, panels, M, D, lamda);
        prepTime = (dsecnd() - sInit);

        // GD
        initW(W,D);
        sInit=dsecnd();

GD:
       printf("start the process \n");
        for (p=0;p<panels;p++){
          I = (float *) malloc(D*D*sizeof(float));
          memset(I,NULL,D*D*sizeof(float));
       // initI(I,D);
         float *Wtmp = (float*)malloc(D*sizeof(float));
         memset(Wtmp,NULL,D*sizeof(float));
         float *XZ = (float*) malloc(D*D*sizeof(float)); //holder of matrix XZ
         memset(XZ,NULL,D*D*sizeof(float));
         float *XZ_r=(float*) malloc(D*D*sizeof(float));
         memset(XZ_r,NULL,D*D*sizeof(float));
         float *XZ_T=(float*) malloc(total_process*D*D*sizeof(float));
         memset(XZ_T,NULL,total_process*D*D*sizeof(float));
            for (i = 1; i < total_process; i++) {
                k = p*panelSz + PchunkSz*(i-1);
                //k = chunkSz*(i-1);
                for(j=0;j<D;j++)
                  printf("Z[%d]=%8.4f /n", j, Z[k*D+j]);
                MPI_Send(NULL, 0, MPI_CHAR, i, X_TAG, MPI_COMM_WORLD);
                MPI_Send(&X[k*D], PchunkSz*D, MPI_FLOAT, i, X_TAG, MPI_COMM_WORLD);
                MPI_Send(&Z[k*D], PchunkSz*D, MPI_FLOAT, i, Z_TAG, MPI_COMM_WORLD);
              // MPI_Send(XZ, D*D, MPI_FLOAT, i, XZ_TAG, MPI_COMM_WORLD);
            }
           // printf("send the data  \n");
            for (i = 1; i < total_process; i++) {
                //collect  data, do the sum and update W; 
                MPI_Recv(XZ_r, D*D, MPI_FLOAT, i, XZ_TAG, MPI_COMM_WORLD, &status);
                //Do vector sum through all nodes.
                 for(j=0;j<D*D;j++){
                
                      XZ_T[(i-1)*D*D+j] +=XZ_r[j];
                  }
          }
          initI(I,D);
         // for(j=0;j<D*D;j++){
         //    I[j]=0.0;
         //  // printf(" I[%d]=%8.4f \n", j, I[j]);
         // }

         // for(j=0;j<D;j++)
         //    I[j*D+j]=1.0;

          for (i = 1; i < total_process; i++) {
                for(j=0;j<D*D;j++){
                 //printf("begin, I[%d]=%8.4f \n", j, I[j]);
                XZ[j] +=XZ_T[(i-1)*D*D+j];
               // I[j] = I[j]-  XZ_T[(i-1)*D*D+j];
               //  printf("first,  XZ_T[%d]=%8.4f \n", j,XZ_T[(i-1)*D*D+j] ); 
                // printf("second, I[%d]=%8.4f \n", j, I[j]);
                 }
            }

         for(i=0;i<D;i++){
            for(j=0;j<D;j++){
            I[i*D+j]= I[i*D+j] -XZ[i*D+j];
           // printf("first, XZ[%d]=%8.4f \n", i*D+j, XZ[i*D+j]);
           // printf("second, I[%d]=%8.4f \n", i*D+j, I[i*D+j]);
         }
        }
 
            cblas_sgemv(CblasRowMajor, CblasNoTrans, D, D, 1, I, D,  W, 1, 0,Wtmp, 1);
            cblas_scopy(D, Wtmp, 1, W, 1);
            cblas_saxpy(D,1,&(B[p*D]),1,W,1);
           // printf("finish one panel  \n");         
            free(Wtmp);
            free(XZ);
            free(XZ_r);
            free(XZ_T);
            free(I);
        }

        gdTime = (dsecnd() - sInit);

        err=calErr(X, Ypred, Ytmp, Y, W, M, D);
        float err_diff = fabsf(err - prev_err);

        //FIXME
        if (err_diff < 0.001) {
            printf("ttlTime=%.5f\t prepTime=%.5f\t gdTime=%.5f\t error=%.5f\n", 
                  (gdTime+prepTime)*1000, prepTime*1000, gdTime*1000, err);
            rep--;
            if (rep == 0) goto finish;
            else goto start;
        }

        prev_err = err;
        goto GD;
   // }

finish:

    for (i = 1; i < total_process; i++) {
        MPI_Send(NULL, 0, MPI_CHAR, i, DIETAG, MPI_COMM_WORLD);
    }

    if (W_tmp) free(W_tmp);
    if (X) free(X);
    if (Z) free(Z);
    if (B) free(B);
   // if (I) free(I);
    if (W) free(W);
    if (Y) free(Y);
    if (Ytmp) free(Ytmp);
    if (Ypred) free(Ypred);

}

void slave() {
    //////////////
    int jj,i,j;
    int p;
    float temp;
   // float *XZ_r = (float*) malloc(D*D*sizeof(float)); 
    float *XZ_temp = (float*) malloc(D*D*sizeof(float)); 
    memset(XZ_temp,NULL,D*D*sizeof(float));
    float *X_r = (float *) malloc(PchunkSz*D*sizeof(float));
    memset(X_r,NULL,D*PchunkSz*sizeof(float));
    float *Z_r = (float *) malloc(PchunkSz*D*sizeof(float));
    memset(Z_r,NULL,D*PchunkSz*sizeof(float));
    float *Z_temp = (float *) malloc(D*sizeof(float));
    memset(Z_temp,NULL,D*sizeof(float));
    float *X_temp = (float *) malloc(D*sizeof(float));
    memset(X_temp,NULL,D*sizeof(float));

    MPI_Status status;

   // printf("slave works \n"); 
   while(1) {

        MPI_Recv(NULL, 0, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == DIETAG) {
            goto s_end;

        } else {
            MPI_Recv(X_r,  PchunkSz*D, MPI_FLOAT, 0, X_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(Z_r, PchunkSz*D, MPI_FLOAT, 0, Z_TAG, MPI_COMM_WORLD, &status);
           // MPI_Recv(XZ_s, D*D, MPI_FLOAT, 0, XZ_TAG, MPI_COMM_WORLD, &status);
            
         //   initG(W_t, D);
         //   for (jj=0; jj<chunkSz;jj++){
         //       float zw = cblas_sdot(D, &(Z_s[jj*D]), 1, W_s, 1);
         //       cblas_saxpy(D,-1*zw,&X_s[jj*D], 1, W_t, 1);
         //   }
        //   for(jj=0;jj<PchunkSz;jj++){
        //     for(i=0;i<D;i++){
        //       for(j=0;j<D;j++){
        //          temp=0;
        //          temp +=X_temp[jj*D+i]*Z_temp[jj*D+j];
        //       }
        //     XZ_temp[i*D+j] +=temp;
        //     }
        //                       
        //   }
            for(jj=0;jj<PchunkSz;jj++){
               cblas_scopy(D, &X_r[jj*D], 1, &X_temp[0], 1);
               cblas_scopy(D, &Z_r[jj*D], 1, &Z_temp[0],1);
                for(i=0;i<D;i++){
                  for(j=0;j<D;j++){
                    temp =0;
                   // for(l=0;l<1;l++){
                    temp += X_temp[i]*Z_temp[j];
                   // }
                    XZ_temp[i*D+j] +=temp ;
                   // printf("temp= %8.4f,   \n", temp);
                  }  
                }
               for(i=0;i<D;i++){
                 X_temp[i]=0;
                 Z_temp[i]=0;
               }
            }
            
            MPI_Send(XZ_temp, D*D, MPI_FLOAT, 0, XZ_TAG, MPI_COMM_WORLD);
            //printf("this slave finish works \n");
           for(i=0;i<D*D;i++){
           XZ_temp[i] =0;
           
           }
       }

    }

s_end:

    //if (W_) free(W_s);
    if (XZ_temp) free(XZ_temp);
    if (X_temp) free(X_temp);
    if (Z_temp) free(Z_temp);

}
// X: a MxD matrix, Y: a M vector, W: a M vector
// W0: a M vector
int main(int argc, char ** argv){
    if (argc>1 && argv[1][0]=='h') {
        printf ("Usage: parSymSGD M D T C lamda r\n");
        printf ("  M: number of data points, D: dimensions, T: time iterations, C: cores;\n");
        printf ("  lamda: learning rate, r: panel size in unit of C.\n");
        return 1;
    }


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_num);
    MPI_Comm_size(MPI_COMM_WORLD, &total_process);

    //read params from command line.
    M = argc>1?atoi(argv[1]):32;
    D = argc>2?atoi(argv[2]):4;
    T = argc>3?atoi(argv[3]):10;
    C = argc>4?atoi(argv[4]):4;
    lamda = argc>5?atof(argv[5]):0.01;
    r = argc>6?atoi(argv[6]):1;

    panelSz = C*r;
    panels = M/panelSz;
    PchunkSz = panelSz/(total_process-1);
    //chunkPreSz = panels/(total_process-1);

    if(process_num == 0) {
        master();
    } else {
        slave();
    }
 
    MPI_Finalize();
    return 0;
}

void initData(float*Wreal, float*W, float*X, float*Y, int M, int D){
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
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, D, alpha, X, D, Wreal, 1, 1, Y, 1);
    ///dump1("Y (real)", Y, M);
}

//call by master
void preprocessPar(float *X, float*Y, float *Z, float *B, int panelSz, int panels, int M, int D, float lamda){

    int p,i,k;
    //float *Z_p = (float *) malloc(chunkPreSz*panelSz*D*sizeof(float));
    //float *B_p = (float *) malloc(chunkPreSz*D*sizeof(float));

    //MPI_Status status;

    initG(B, panels*D);
    //FIXME:set Z to 0 may be buggy.
    initG(Z, M*D);


    float *XX = (float*) malloc(panelSz*panelSz*sizeof(float)); //holder of X[j]X[j] for one panel

    //int chunkSz=panels/nt;
//#pragma omp parallel for private(i,p)
    for (p=0;p<panels;p++){
        ///printf("preprocess panel %d...\n", p);
        ///printf("%f\n", X[p*panelSz*D]);
        preprocess1(panelSz, D, XX, &(X[p*panelSz*D]), &(Y[p*panelSz]), lamda, &(Z[p*panelSz*D]), &(B[p*D]));
    }
    if (XX) free(XX);

    /*for (i = 1; i < total_process; i++) {
        //k = p*panelSz + chunkSz*(i-1);
        k = chunkPreSz*panelSz*(i-1);

        MPI_Send(NULL, 0, MPI_CHAR, i, PREPARETAG, MPI_COMM_WORLD);
        MPI_Send(&X[k*D], chunkPreSz*panelSz*D, MPI_FLOAT, i, X_TAG, MPI_COMM_WORLD);
        MPI_Send(&Y[k], chunkPreSz*panelSz, MPI_FLOAT, i, Y_TAG, MPI_COMM_WORLD);
    }


    for (i = 1; i < total_process; i++) {
        k = i -1;
        //collect  data, do the sum and update W; 
        MPI_Recv(Z_p, chunkPreSz*panelSz*D, MPI_FLOAT, i, Z_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(B_p, chunkPreSz*D, MPI_FLOAT, i, B_TAG, MPI_COMM_WORLD, &status);
        //Do vector sum through all nodes.
        cblas_saxpy(chunkPreSz*panelSz*D,1,Z_p,1,&(Z[k*chunkPreSz*panelSz*D]),1);
        cblas_saxpy(chunkPreSz*panelSz,1,B_p,1,&(B[k*chunkPreSz*D]),1);

    }
    if (Z_p) free(Z_p);
    if (B_p) free(B_p);*/
}

//call by master
void preprocess1(int panelSz, int D, float *XX, float*X, float*Y, float lamda, float*Z, float*B){
    // panelSz: the number of points to process in each round
    int i,j;
    // step 1: compute all X[i]'*X[j] (i>j)
    for (i=panelSz-1;i>0;i--)
        for (j=i-1;j>=0;j--){
            XX[i*panelSz+j] = cblas_sdot(D, &(X[i*D]), 1, &(X[j*D]), 1);
            ///printf("XX[%d]=%8.4f, X[%d]=%8.4f, X[%d]=%8.4f\n", i*panelSz+j, XX[i*panelSz+j], i*D, X[i*D], j*D, X[j*D]);
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

void initI(float*I,int D){
int i;
for(i=0;i<D*D;i++)
  I[i] = 0.0;
for(i=0;i<D;i++)
  I[i*D+i] = 1.0;


}


void dump1(char*s, float*V, int D){
    //printf("%d, %s:\n",rank, s);
    printf("%s:\n", s);
    int i;
    for (i=0;i<D;i++)
        printf("%8.4f  ", V[i]);
    printf("\n--------------\n");
}

void initG(float*G, int D){
    // initialize W
    int i;
    for (i=0;i<D;i++){ 
        G[i] = 0;
    }
}

