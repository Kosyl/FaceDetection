#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_LEN 256
#define LEN 50

char * read_line(FILE *);

int main(int argc, char *argv[]) {
	FILE *fp , *fp_t;
	int i = 0;
	int j = 0;
	int k = 0;
	char c = '0';
	int parsing = 0;
	int parsing_line = 0;
	int parsing_stage = 0;
	int parsing_sizes = 0;
	char temp_char[LEN];
	char *temp_f[3];
	char *temp_i1[5];
	char *temp_i2[5];
	int k_i = 0;
	int sizes[21];
	for (i = 0; i < 3; i++)
		temp_f[i] = (char *)malloc(sizeof(char) * LEN);
	for (i = 0; i < 5; i++)
		temp_i1[i] = (char *)malloc(sizeof(char) * LEN);
	for (i = 0; i < 5; i++)
		temp_i2[i] = (char *)malloc(sizeof(char) * LEN);

	if ((fp = fopen("FaceHaarCascade.cs", "r")) == NULL) {
		perror("Blad odczytu");
		exit(-1);
		}
	if ((fp_t = fopen("code.cpp", "w")) == NULL) {
		perror("Blad odczytu");
		exit(-1);
		}
	i = 0;
	j = 0;
	char *buff = (char *)malloc(sizeof(char) * MAX_LEN);
	while(fgets(buff, MAX_LEN, fp)) {
		if (strstr(buff, "stage ="))
			parsing_sizes = 1;
		if (strstr(buff, "nodes.Add"))
			i++;
		if (strstr(buff, "stage.Trees")) {
			sizes[j++] = i;
			parsing_sizes = 0;
			i = 0;
		}
	}
	i = 0;
	j = 0;
	fseek(fp, 0, SEEK_SET);
	while(fgets(buff, MAX_LEN, fp)) {
		//printf("%s", buff);
		if (strstr(buff, "stage =")) {
		/* threshold */
			for (i = 0, j = 0; i < strlen(buff); i++) {
				//printf("%c", buff[i]);
				if ((parsing == 0) && isdigit(buff[i])) {
					parsing = 1;
					temp_char[j++] = buff[i];
				} else
				if (parsing == 1) {
					if (buff[i] == ')') {
						parsing = 0;
						temp_char[j++] = '\0';
					}	
					temp_char[j++] = buff[i];
				}

			}
		printf("%s \n", temp_char);
		fprintf(fp_t, "\nstages[%d].threshold = %s;\n", k, temp_char);
		fprintf(fp_t, "stages[%d].haarAreas = new HaarArea*[%d];\n", k, sizes[k]);
		fprintf(fp_t, "stages[%d].numAreas = %d;\n", k, sizes[k]);
		k++;
		parsing_stage = 1;
		k_i = 0;
		} 
		/* wspolczynniki */ 
		else
		if (strstr(buff, "nodes.Add") && (parsing_stage == 1)) {
			k_i++;
			int fin = 0;
			for (i = 0, j = 0; i < strlen(buff); i++) {
				if ((parsing_line == 0) && (isdigit(buff[i]) || (buff[i] == '-'))) {
					parsing_line = 1;
					temp_f[0][j++] = buff[i];
				} else
				if (parsing_line == 1) {
					if (buff[i] == ',')
						fin++;
					if (fin == 3) {
						parsing_line = 0;
						//parsing_stage = 0;
						temp_f[0][j++] = '\0';
					}	
					temp_f[0][j++] = buff[i];
				}
			}	
		printf("%s ", temp_f[0]);
		parsing_line = 0;

		char *int_c = strstr(buff, "new int[] { ");
		int l = 0;
		int p = 0;
		int m = 0;
		int tru = 0;
		while (*int_c != '\n') {
			if ((isdigit(*int_c)) || (*int_c == ',') || (*int_c == '-')) 
				temp_i1[0][l++] = *int_c;
			int_c++;
		}
		temp_i1[0][l] = '\0';
		printf("%s \n", temp_i1[0]);
		
		l = 0;
		for (i = 0, j = 0; i < strlen(temp_i1[0]); i++) {
			if (temp_i1[0][i] == ',')
				p++;
			if (p < 5) {
				if ((isdigit(temp_i1[0][i])) || (temp_i1[0][i] == ',') || (temp_i1[0][i] == '-'))
				temp_i2[0][j++] = temp_i1[0][i];
			} else if ((p >= 5) && (p < 10)) {
				if ((temp_i1[0][i] == ',') && (p == 5)); else
				if ((isdigit(temp_i1[0][i])) || (temp_i1[0][i] == ',') || (temp_i1[0][i] == '-'))
				temp_i2[1][l++] = temp_i1[0][i];			
			} else if (p >= 10) {
				tru = 1;
				if ((temp_i1[0][i] == ',') && (p == 10)); else
				if ((isdigit(temp_i1[0][i])) || (temp_i1[0][i] == ',') || (temp_i1[0][i] == '-'))
				temp_i2[2][m++] = temp_i1[0][i];
				}

		}
		temp_i2[0][j] = '\0';
		temp_i2[1][l] = '\0';
		temp_i2[2][m] = '\0';
		if (tru == 0)
			 fprintf(fp_t, "stages[%d].haarAreas[%d] = new HaarArea(%s, HaarRectangle(%s), HaarRectangle(%s));\n", k-1, k_i - 1, temp_f[0], temp_i2[0], temp_i2[1]);
		else
			fprintf(fp_t, "stages[%d].haarAreas[%d] = new HaarArea(%s, HaarRectangle(%s), HaarRectangle(%s), HaarRectangle(%s));\n", k-1, k_i - 1, temp_f[0], temp_i2[0], temp_i2[1], temp_i2[2]);	
		}
	}
	fclose(fp);
	return 0;
	}
