#include <stdio.h>
#include <stdlib.h>
#include "nrutil.h"
#include <time.h>
#include <math.h>
#include "nrutil.c"

int main()
{

int i,j,*s,ancho,N=60,LARGO=N*N,TIEMPO_TOMA=1000*LARGO*7;
double *frecuency,aa=1.02;
double desde,hasta,primer,segundo,decimal,desdeoriginal,desdemas1,geomean,valorc,normalizc,rNREALIZ;

FILE *fpentrada;
fpentrada=fopen("input_binning.dat","r");
FILE *fpsalida;
fpsalida=fopen("output_binning.dat","w");

s=ivector(1,LARGO);
frecuency=dvector(1,2*LARGO);

for (i=1;i<=2*LARGO;i++){frecuency[i]= 0;}

rNREALIZ=(double)TIEMPO_TOMA;
normalizc = (long double)(LARGO*rNREALIZ);

for (i=1;i<=LARGO;i++){
fscanf(fpentrada,"%d %lf",&s[i],&frecuency[i]);}

for (i=1;i<=LARGO;i++){frecuency[i]= frecuency[i] * normalizc;}

i=0;
while( (pow(aa,i)) <= (double) LARGO){
                         // printf("%d desde %lf a %lf \n",i,pow(aa,i),pow(aa,i+1));
   desde=(pow(aa,i));
   hasta=(pow(aa,i+1));
   if (floor(desde) == desde ){desde=desde - 0.1;}
   if (floor(hasta) == hasta ){hasta=hasta + 0.1;}

   decimal = modf (desde, &primer);
   decimal = modf (hasta, &segundo);

   ancho=(int)segundo-(int)primer;

valorc = 0.;

if (ancho>0) {        
      desdeoriginal=(pow(aa,i));
      desdemas1=desde+1;
      decimal = modf (desdemas1, &primer);
      decimal = modf (hasta, &segundo);
      if (segundo>=LARGO) {segundo=LARGO;  ancho=(int)segundo-(int)primer;}  
      geomean=sqrt(primer*segundo);    // printf("ancho= %d, desde %lf a %lf, in x=%17.10e \n",ancho,primer,segundo,geomean); 

for (j=primer;j<=segundo;j++){ valorc=valorc+frecuency[j]; }
valorc=valorc/((long double)(normalizc*ancho)); 
fprintf(fpsalida,"%17.10e %17.10e \n",geomean,valorc); 
fflush(fpsalida);
}
i++;
}

}
