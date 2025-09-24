#include <stdio.h>
#include <math.h>

int main (){
float num, fah;
    printf("input temperature in degree celsius :");
    scanf("%f" , &num);
    fah = (1.8 * num) + 32;
    printf("temperature in fahrenheit : %f \n" , fah);
return 0;
}
