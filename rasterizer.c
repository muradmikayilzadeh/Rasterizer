/*
    Name: Rasterizer Program and Animation
    Author: Murad Mikayilzade
    Animation link: https://www.youtube.com/shorts/W5ZgaESP2LU
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NORMALS

/*
Image struct with members of
    1. width: width of generated image
    2. height: height of the generated image
    3. channels: number of image color channels(3 for r,g,b)
    4. data: array of unsigned chars to keep rgb values for relevant pixels.
*/ 
struct Image {
    int width;
    int height;
    int channels;
    unsigned char *data;
    double *depthBuffer;
};

double color_interpolations[3][2];
double shade_interpolations[2];

double C441(double f)
{
    return ceil(f-0.00001);
}

double F441(double f)
{
    return floor(f+0.00001);
}

typedef struct
{
    double          A[4][4];     // A[i][j] means row i, column j
} Matrix;


void
PrintMatrix(Matrix m)
{
    for (int i = 0 ; i < 4 ; i++)
    {
        printf("(%.7f %.7f %.7f %.7f)\n", m.A[i][0], m.A[i][1], m.A[i][2], m.A[i][3]);
    }
}

double 
DotProduct(double a[3], double b[3]) {
  double result = 0.0;
  for (int i = 0; i < 3; i++) {
    result += a[i] * b[i];
  }
  return result;
}

double 
cot(double x) {
  return 1.0 / tan(x);
}


Matrix
ComposeMatrices(Matrix M1, Matrix M2)
{
    Matrix m_out;
    for (int i = 0 ; i < 4 ; i++)
        for (int j = 0 ; j < 4 ; j++)
        {
            m_out.A[i][j] = 0;
            for (int k = 0 ; k < 4 ; k++)
                m_out.A[i][j] += M1.A[i][k]*M2.A[k][j];
        }
    return m_out;
}

void 
TransformPoint(Matrix m, const double *ptIn, double *ptOut)
{  
    ptOut[0] = ptIn[0]*m.A[0][0]
             + ptIn[1]*m.A[1][0]
             + ptIn[2]*m.A[2][0]
             + ptIn[3]*m.A[3][0];
    ptOut[1] = ptIn[0]*m.A[0][1]
             + ptIn[1]*m.A[1][1]
             + ptIn[2]*m.A[2][1]
             + ptIn[3]*m.A[3][1];
    ptOut[2] = ptIn[0]*m.A[0][2]
             + ptIn[1]*m.A[1][2]
             + ptIn[2]*m.A[2][2]
             + ptIn[3]*m.A[3][2];
    ptOut[3] = ptIn[0]*m.A[0][3]
             + ptIn[1]*m.A[1][3]
             + ptIn[2]*m.A[2][3]
             + ptIn[3]*m.A[3][3];
}


typedef struct
{
    double          near, far;
    double          angle;
    double          position[3];
    double          focus[3];
    double          up[3];
} Camera;

typedef struct 
{
    double lightDir[3]; // The direction of the light source
    double Ka;           // The coefficient for ambient lighting.
    double Kd;           // The coefficient for diffuse lighting.
    double Ks;           // The coefficient for specular lighting.
    double alpha;        // The exponent term for specular lighting.
} LightingParameters;

double SineParameterize(int curFrame, int nFrames, int ramp)
{  
    int nNonRamp = nFrames-2*ramp;
    double height = 1./(nNonRamp + 4*ramp/M_PI);
    if (curFrame < ramp)
    {
        double factor = 2*height*ramp/M_PI;
        double eval = cos(M_PI/2*((double)curFrame)/ramp);
        return (1.-eval)*factor;
    }
    else if (curFrame > nFrames-ramp)
    {        
        int amount_left = nFrames-curFrame;
        double factor = 2*height*ramp/M_PI;
        double eval =cos(M_PI/2*((double)amount_left/ramp));
        return 1. - (1-eval)*factor;
    }        
    double amount_in_quad = ((double)curFrame-ramp);
    double quad_part = amount_in_quad*height;
    double curve_part = height*(2*ramp)/M_PI;
    return quad_part+curve_part;
} 

Camera       
GetCamera(int frame, int nframes)
{            
    double t = SineParameterize(frame, nframes, nframes/10);
    Camera c;
    c.near = 5;
    c.far = 200;
    c.angle = M_PI/6;
    c.position[0] = 40*sin(2*M_PI*t);
    c.position[1] = 40*cos(2*M_PI*t);
    c.position[2] = 40;
    c.focus[0] = 0; 
    c.focus[1] = 0; 
    c.focus[2] = 0;
    c.up[0] = 0;    
    c.up[1] = 1;    
    c.up[2] = 0;    
    return c;       
}

Matrix 
GetViewTransform(Camera *c)
{
    Matrix rv;

    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++)
            rv.A[i][j] = 0;
    }
    rv.A[0][0] = cot(c->angle/2);
    rv.A[1][1] = cot(c->angle/2);

    rv.A[2][2] = (c->far + c->near)/(c->far - c->near);
    rv.A[2][3] = -1;

    rv.A[3][2] = (2*c->far*c->near)/(c->far-c->near);

    return rv;
}

Matrix
GetCameraTransform(Camera *c)
{   

    // O - focus
    double w[3] = {
        (c->position[0] - c->focus[0]),(c->position[1] - c->focus[1]), (c->position[2] - c->focus[2])
    };
    // Normalize w vector
    double w_normal = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    w[0] = w[0]/w_normal;
    w[1] = w[1]/w_normal;
    w[2] = w[2]/w_normal;


    double u[3] = {
        c->up[1] * w[2] - c->up[2] * w[1],
        c->up[2] * w[0] - c->up[0] * w[2],
        c->up[0] * w[1] - c->up[1] * w[0]
    };
    // Normalize u vector
    double u_normal = sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
    u[0] = u[0]/u_normal;
    u[1] = u[1]/u_normal;
    u[2] = u[2]/u_normal;

    double v[3] = {
        w[1] * u[2] - w[2] * u[1],
        w[2] * u[0] - w[0] * u[2],
        w[0] * u[1] - w[1] * u[0]
    };  
    // Normalize v vector
    double v_normal = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    v[0] = v[0]/v_normal;
    v[1] = v[1]/v_normal;
    v[2] = v[2]/v_normal;

    // (0,0,0) - O
    double t[3] = {
        (0 - c->position[0]), (0 - c->position[1]), (0 - c->position[2])
    };

    double cameraTransformMatrix[4][4] = {
        {u[0], v[0], w[0], 0},
        {u[1], v[1], w[1], 0},
        {u[2], v[2], w[2], 0},
        {DotProduct(u, t), DotProduct(v, t), DotProduct(w, t), 1}
    };

    Matrix rv;
    
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++)
            rv.A[i][j] = cameraTransformMatrix[i][j];
    }

    return rv;
}

Matrix
GetDeviceTransform(Camera *c, int n, int m)
{   
    Matrix rv;

    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++)
            rv.A[i][j] = 0;
    }

    rv.A[0][0] = n/2;
    rv.A[3][0] = n/2;

    rv.A[1][1] = m/2;
    rv.A[3][1] = m/2;

    rv.A[2][2] = 1;
    rv.A[3][3] = 1;

    return rv;
}


LightingParameters 
GetLighting(Camera c)
{
    LightingParameters lp;
    lp.Ka = 0.3;
    lp.Kd = 0.7;
    lp.Ks = 2.8;
    lp.alpha = 50.5;
    lp.lightDir[0] = c.position[0]-c.focus[0];
    lp.lightDir[1] = c.position[1]-c.focus[1];
    lp.lightDir[2] = c.position[2]-c.focus[2];
    double mag = sqrt(lp.lightDir[0]*lp.lightDir[0]
                    + lp.lightDir[1]*lp.lightDir[1]
                    + lp.lightDir[2]*lp.lightDir[2]);
    if (mag > 0)
    {
        lp.lightDir[0] /= mag;
        lp.lightDir[1] /= mag;
        lp.lightDir[2] /= mag;
    }

    return lp;
}

typedef struct
{
   double         X[3];
   double         Y[3];
   double         Z[3];
   double         color[3][3]; // color[2][0] is for V2, red channel
#ifdef NORMALS
   double         normals[3][3]; // normals[2][0] is for V2, x-component
   double         shading[3]; // shading[0] is shading value of 0th vertex
#endif
} Triangle;

typedef struct
{
   int numTriangles;
   Triangle *triangles;
} TriangleList;

char *
Read3Numbers(char *tmp, double *v1, double *v2, double *v3)
{
    *v1 = atof(tmp);
    while (*tmp != ' ')
       tmp++;
    tmp++; /* space */
    *v2 = atof(tmp);
    while (*tmp != ' ')
       tmp++;
    tmp++; /* space */
    *v3 = atof(tmp);
    while (*tmp != ' ' && *tmp != '\n')
       tmp++;
    return tmp;
}

TriangleList *
Get3DTriangles()
{
   FILE *f = fopen("ws_tris.txt", "r");
   if (f == NULL)
   {
       fprintf(stderr, "You must place the ws_tris.txt file in the current directory.\n");
       exit(EXIT_FAILURE);
   }
   fseek(f, 0, SEEK_END);
   int numBytes = ftell(f);
   fseek(f, 0, SEEK_SET);
   if (numBytes != 3892295)
   {
       fprintf(stderr, "Your ws_tris.txt file is corrupted.  It should be 3892295 bytes, but you have %d.\n", numBytes);
       exit(EXIT_FAILURE);
   }

   char *buffer = (char *) malloc(numBytes);
   if (buffer == NULL)
   {
       fprintf(stderr, "Unable to allocate enough memory to load file.\n");
       exit(EXIT_FAILURE);
   }
   
   fread(buffer, sizeof(char), numBytes, f);

   char *tmp = buffer;
   int numTriangles = atoi(tmp);
   while (*tmp != '\n')
       tmp++;
   tmp++;
 
   if (numTriangles != 14702)
   {
       fprintf(stderr, "Issue with reading file -- can't establish number of triangles.\n");
       exit(EXIT_FAILURE);
   }

   TriangleList *tl = (TriangleList *) malloc(sizeof(TriangleList));
   tl->numTriangles = numTriangles;
   tl->triangles = (Triangle *) malloc(sizeof(Triangle)*tl->numTriangles);

   for (int i = 0 ; i < tl->numTriangles ; i++)
   {
       for (int j = 0 ; j < 3 ; j++)
       {
           double x, y, z;
           double r, g, b;
           double normals[3];
/*
 * Weird: sscanf has a terrible implementation for large strings.
 * When I did the code below, it did not finish after 45 minutes.
 * Reading up on the topic, it sounds like it is a known issue that
 * sscanf fails here.  Stunningly, fscanf would have been faster.
 *     sscanf(tmp, "(%lf, %lf), (%lf, %lf), (%lf, %lf) = (%d, %d, %d)\n%n",
 *              &x1, &y1, &x2, &y2, &x3, &y3, &r, &g, &b, &numRead);
 *
 *  So, instead, do it all with atof/atoi and advancing through the buffer manually...
 */
           tmp = Read3Numbers(tmp, &x, &y, &z);
           tmp += 3; /* space+slash+space */
           tmp = Read3Numbers(tmp, &r, &g, &b);
           tmp += 3; /* space+slash+space */
           tmp = Read3Numbers(tmp, normals+0, normals+1, normals+2);
           tmp++;    /* newline */

           tl->triangles[i].X[j] = x;
           tl->triangles[i].Y[j] = y;
           tl->triangles[i].Z[j] = z;
           tl->triangles[i].color[j][0] = r;
           tl->triangles[i].color[j][1] = g;
           tl->triangles[i].color[j][2] = b;
#ifdef NORMALS
           tl->triangles[i].normals[j][0] = normals[0];
           tl->triangles[i].normals[j][1] = normals[1];
           tl->triangles[i].normals[j][2] = normals[2];
#endif
       }
   }

   free(buffer);
   return tl;
}

/*
Helper function that accepts arguments of:
    1. struct Image *img: reference to the image struct
    2. colMin: starting index of the pixel in x-axis
    3. rolMax: ending index of the pixel in x-axis
    4. rowMin: starting index of the pixel in y-axis
    5. rowMax: ending index of the pixel in y-axis
    6. unsigned char R, G, B: RGB values for this particular pixel

    The function takes the above parametres and paints the pixels in between colMin-colMax and rowMin-rowMax 
    with the given RGB color values. 
*/
void assign_pixels(struct Image *img, int colMin, int colMax, int rowMin, int rowMax, unsigned char R, unsigned char G, unsigned char B){
    
    int index;
    for(int i = colMin; i <= colMax; i++){
        for(int j = rowMin; j <= rowMax; j++){
            index = (j * img->width + i) * img->channels;
            img->data[index] = R;
            img->data[index + 1] = G;
            img->data[index + 2] = B;
        }
    }

}

/*
Helper function that creates .pnm file and fill it with file headings and
    data content.
*/
void write_image(struct Image *img, char *fileName){
    
    // Create PNM file
    FILE *fp = fopen(fileName, "w");

    // Write PNM file header
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", img->width, img->height);
    fprintf(fp, "255\n");
    
    // Write PNM file body
    fwrite(img->data, sizeof(unsigned char), img->width * img->height * img->channels, fp);

    // Close file
    fclose(fp);
}

double interpolation(double fA, double fB, double A, double B, double X){
    if(A == B)
        return fA;
    
    double t = (X - A) / (B - A);
    double Y = fA + t * (fB - fA);

    return Y;
}

double lerp_shading(float start, float end, float t) {
    return start + t * (end - start);
}

void normalize(double vector[3]){
    double norm = sqrt(vector[0] * vector[0] +
                        vector[1] * vector[1] +
                        vector[2] * vector[2]);

    vector[0] /= norm;
    vector[1] /= norm;
    vector[2] /= norm;
}

double calculate_shading(LightingParameters *lp, double *viewDirection, Triangle *triangle, int vertexIndex)
{
    double diffuse, specular, lighting;
    double normal[3] = {
        triangle->normals[vertexIndex][0], triangle->normals[vertexIndex][1], triangle->normals[vertexIndex][2]
    };
    normalize(normal);

    double light_direction[3] = {
        lp->lightDir[0], 
        lp->lightDir[1], 
        lp->lightDir[2]
    };
    normalize(light_direction);

    double LdotN = DotProduct(light_direction, normal);

    diffuse = lp->Kd * fmax(0, LdotN);
    double reflection_vector[3] = {
        2 * LdotN * normal[0] - light_direction[0],
        2 * LdotN * normal[1] - light_direction[1],
        2 * LdotN * normal[2] - light_direction[2]
    };
    normalize(reflection_vector);
    double RdotV = DotProduct(reflection_vector, viewDirection);

    specular = lp->Ks * pow(fmax(0,RdotV), lp->alpha);

    lighting = lp->Ka + diffuse + specular;

    return lighting;
    
}

double color_interpolation(Triangle* triangle, int topVertex, int leftVertex, int rightVertex, int i){
    color_interpolations[0][0] = interpolation(triangle->color[leftVertex][0], triangle->color[topVertex][0], triangle->Y[leftVertex], triangle->Y[topVertex], i);
    color_interpolations[0][1] = interpolation(triangle->color[rightVertex][0], triangle->color[topVertex][0], triangle->Y[rightVertex], triangle->Y[topVertex], i);
    
    color_interpolations[1][0] = interpolation(triangle->color[leftVertex][1], triangle->color[topVertex][1], triangle->Y[leftVertex], triangle->Y[topVertex], i);
    color_interpolations[1][1] = interpolation(triangle->color[rightVertex][1], triangle->color[topVertex][1], triangle->Y[rightVertex], triangle->Y[topVertex], i);

    color_interpolations[2][0] = interpolation(triangle->color[leftVertex][2], triangle->color[topVertex][2], triangle->Y[leftVertex], triangle->Y[topVertex], i);
    color_interpolations[2][1] = interpolation(triangle->color[rightVertex][2], triangle->color[topVertex][2], triangle->Y[rightVertex], triangle->Y[topVertex], i);
}

double shade_interpolation(Triangle* triangle, int topVertex, int leftVertex, int rightVertex, int i){
    shade_interpolations[0] = interpolation(triangle->shading[leftVertex], triangle->shading[topVertex], triangle->Y[leftVertex], triangle->Y[topVertex], i);
    shade_interpolations[1] = interpolation(triangle->shading[rightVertex], triangle->shading[topVertex], triangle->Y[rightVertex], triangle->Y[topVertex], i);
}

double vertex_shading(Triangle* triangle, LightingParameters* lp, Camera* c){

    for (int i = 0; i < 3; i++) {

        double viewDirection[3] = {
            c->position[0] - triangle->X[i],
            c->position[1] - triangle->Y[i],
            c->position[2] - triangle->Z[i]
        };
        normalize(viewDirection);

        triangle->shading[i] = calculate_shading(lp, viewDirection, triangle, i);
    }
}

void rasterize(struct Image* img, Triangle* t, Triangle* oldTriangle, LightingParameters* lp, Camera* c){ 

    for(int i = 0; i < 2; i++){
        int workOnTopPart = (i==0 ? 1 : 0);

        int topVertex = -1;
        int leftVertex = -1;
        int rightVertex = -1;
        
        // Figuring out Top, left and right vertices.
        // 0=>top; 1=>left; 2=>right
        if(workOnTopPart){
            if(t->Y[0] > t->Y[1] && t->Y[0] > t->Y[2]){
                topVertex = 0;
                if(t->X[1] < t->X[2]){
                    leftVertex = 1;
                    rightVertex = 2;
                }else{
                    leftVertex = 2;
                    rightVertex = 1;
                }      
            }
            else if(t->Y[1] > t->Y[0] && t->Y[1] > t->Y[2]){
                topVertex = 1;
                if(t->X[0] < t->X[2]){
                    leftVertex = 0;
                    rightVertex = 2;
                }else{
                    leftVertex = 2;
                    rightVertex = 0;
                }    
            }
            else if(t->Y[2] > t->Y[1] && t->Y[2] > t->Y[0]){
                topVertex = 2;
                if(t->X[1] < t->X[0]){
                    leftVertex = 0;
                    rightVertex = 1;
                }else{
                    leftVertex = 1;
                    rightVertex = 0;
                }    
            }
        }
        else{
            if(t->Y[0] < t->Y[1] && t->Y[0] < t->Y[2]){
                topVertex = 0;
                if(t->X[1] < t->X[2]){
                    leftVertex = 1;
                    rightVertex = 2;
                }else{
                    leftVertex = 2;
                    rightVertex = 1;
                }    
            }
            else if(t->Y[1] < t->Y[0] && t->Y[1] < t->Y[2]){
                topVertex = 1;
                if(t->X[0] < t->X[2]){
                    leftVertex = 0;
                    rightVertex = 2;
                }else{
                    leftVertex = 2;
                    rightVertex = 0;
                }    
            }
            else if(t->Y[2] < t->Y[1] && t->Y[2] < t->Y[0]){
                topVertex = 2;
                if(t->X[1] < t->X[0]){
                    leftVertex = 0;
                    rightVertex = 1;
                }else{
                    leftVertex = 1;
                    rightVertex = 0;
                } 
            }
        }

        // Line equations
        double mL, mR, bL, bR;

        mL = (t->Y[topVertex] - t->Y[leftVertex])/(t->X[topVertex] - t->X[leftVertex]);
        bL = t->Y[topVertex] - mL * t->X[topVertex];
        
        mR = (t->Y[topVertex] - t->Y[rightVertex])/(t->X[topVertex] - t->X[rightVertex]);
        bR = t->Y[topVertex] - mR * t->X[topVertex];  

        if(isinf(mL) || isnan(mL)){
            mL, bL = 0;
        } 

        if(isinf(mR) || isnan(mR)){
            mR, bR = 0;
        } 
        
        double top_scanline,bottom_scanline = -1;

        if (workOnTopPart) {
            if(t->Y[rightVertex] > t->Y[leftVertex]) {
                bottom_scanline = t->Y[rightVertex];
                top_scanline = t->Y[topVertex];
            }else{
                bottom_scanline = t->Y[leftVertex];
                top_scanline = t->Y[topVertex];
            }
        }else{
            if(t->Y[rightVertex] > t->Y[leftVertex]) {
                top_scanline = t->Y[leftVertex];
                bottom_scanline = t->Y[topVertex];
            }else{
                top_scanline = t->Y[rightVertex];
                bottom_scanline = t->Y[topVertex];
            }
        }
        
    
        if(top_scanline >= img->width)
            top_scanline = img->width-1;

        if(bottom_scanline < 0)
            bottom_scanline = 0;
        
        double leftEnd, rightEnd, temp;
        int newRow, index;
        for(int i = C441(bottom_scanline); i <= F441(top_scanline); i+=1){

            if((t->X[topVertex] == t->X[leftVertex] && t->Y[topVertex] != t->Y[leftVertex])){
                leftEnd = t->X[leftVertex];
            }else{
                leftEnd = (i-bL)/mL;
            }

            if(t->X[topVertex] == t->X[rightVertex] && t->Y[topVertex] != t->Y[rightVertex]){
                rightEnd = t->X[rightVertex];
            }else{
                rightEnd = (i-bR)/mR;
            }

            int swapped = 0;
            if(rightEnd < leftEnd){
                swapped = 1;

                temp = rightEnd;
                rightEnd = leftEnd;
                leftEnd = temp;
            }
            
            double zleftEnd = interpolation(t->Z[leftVertex], t->Z[topVertex], t->Y[leftVertex], t->Y[topVertex],i);
            double zrightEnd = interpolation(t->Z[rightVertex], t->Z[topVertex], t->Y[rightVertex], t->Y[topVertex],i);

            color_interpolation(t, topVertex, leftVertex, rightVertex, i);
            shade_interpolation(t, topVertex, leftVertex, rightVertex, i);

            for(int j = C441(leftEnd); j <= F441(rightEnd); j+=1){
            
                double z = ((j - leftEnd)/(rightEnd - leftEnd)) * (zrightEnd - zleftEnd) + zleftEnd;

                // Color and shading interpolations
                double R, G, B, shading_value;
                if(swapped){
                    shading_value = shade_interpolations[0] + ((j - rightEnd)/(leftEnd - rightEnd)) * (shade_interpolations[1] - shade_interpolations[0]);
                    R = color_interpolations[0][0] + ((j - rightEnd)/(leftEnd - rightEnd)) * (color_interpolations[0][1] - color_interpolations[0][0]);
                    G = color_interpolations[1][0] + ((j - rightEnd)/(leftEnd - rightEnd)) * (color_interpolations[1][1] - color_interpolations[1][0]);
                    B = color_interpolations[2][0] + ((j - rightEnd)/(leftEnd - rightEnd)) * (color_interpolations[2][1] - color_interpolations[2][0]);
                }else{
                    shading_value = shade_interpolations[0] + ((j - leftEnd)/(rightEnd - leftEnd)) * (shade_interpolations[1] - shade_interpolations[0]);
                    R = color_interpolations[0][0] + ((j - leftEnd)/(rightEnd - leftEnd)) * (color_interpolations[0][1] - color_interpolations[0][0]);
                    G = color_interpolations[1][0] + ((j - leftEnd)/(rightEnd - leftEnd)) * (color_interpolations[1][1] - color_interpolations[1][0]);
                    B = color_interpolations[2][0] + ((j - leftEnd)/(rightEnd - leftEnd)) * (color_interpolations[2][1] - color_interpolations[2][0]);
                }
                
                newRow = img->height-i-1;
                index = i * img->width + j;

                if(img->depthBuffer[index] < z && j<img->width){
                    if(newRow < img->height && newRow >= 0)
                        assign_pixels(img, j, j, newRow, newRow, C441(255*fmin(1, R*shading_value)), C441(255*fmin(1, G*shading_value)), C441(255*fmin(1, B*shading_value))); 
                    img->depthBuffer[index] = z;                 
                }    

            }
        }
    }
}

void apply_transformation(Triangle *t, Matrix *transformation){

    double vertex_out[4];
    for(int i = 0; i < 3; i++){
        double vertex[4] = {
            t->X[i], t->Y[i], t->Z[i], 1
        };

        TransformPoint(*transformation, vertex, vertex_out);

        t->X[i] = vertex_out[0]/vertex_out[3];
        t->Y[i] = vertex_out[1]/vertex_out[3];
        t->Z[i] = vertex_out[2]/vertex_out[3];
    }

}

void TransformAndRenderTriangles(Camera* c, LightingParameters* lp, TriangleList* tl, struct Image* img){

    Matrix transformation = ComposeMatrices(GetCameraTransform(c),GetViewTransform(c));
    transformation = ComposeMatrices(transformation, GetDeviceTransform(c, img->width, img->height));

    for (int i = 0; i < tl->numTriangles; i++) {
        
        Triangle triangle = tl->triangles[i];

        vertex_shading(&triangle, lp, c);

        Triangle newTriangle = triangle; 
        apply_transformation(&newTriangle, &transformation);

        rasterize(img,&newTriangle, &triangle, lp, c);

    }

}

int main(void)
{

    struct Image img;
    img.width = 1000;
    img.height = 1000;
    img.channels = 3;
    img.data = (unsigned char *)malloc(img.width * img.height * img.channels);
    img.depthBuffer = malloc(img.width * img.height * sizeof(double));

    for (int i = 0; i < img.width * img.height; i++) {
        img.depthBuffer[i] = -1;
    }

    TriangleList *tl = Get3DTriangles();
    // for (int i = 0 ; i < 1000 ; i++) {

        assign_pixels(&img, 0, img.width, 0, img.height, 0, 0, 0); //black
        for (int i = 0; i < img.width * img.height; i++) {
            img.depthBuffer[i] = -1;
        }
        int i = 0;
        Camera c = GetCamera(i, 1000);
        LightingParameters lp = GetLighting(c);
        TransformAndRenderTriangles(&c, &lp, tl, &img);

        char fileName[50];

        sprintf(fileName, "proj1F_frame%04d.pnm",i);

        write_image(&img, fileName);

    // }
    
    return 0;
}
