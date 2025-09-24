#include <stdio.h>
#include <math.h>

int main(){
float num, vol,area;
printf("input radius of sphere :");
scanf("%f", &num);
vol = 1.33 * 3.14 * num * num * num;
area = 4 * 3.14 * num * num;
printf("area of sphere : %f\n",area);
printf("volume of sphere : %f\n" , vol);
return 0;
}
