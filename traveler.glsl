#define TAU 6.283185307
#define PI 3.141592654
#define U(z,w) (z.x < w.x ? z : w)

#define MAT_WING  1.0
#define MAT_BODY  2.0
#define MAT_STAGE 3.0

const int Iterations = 3;

float time;
float beat, kick, hihat, snare;
float stageScale;
mat3 sphereRot, stageRot, stageRot2;
vec3 ray;
vec3 ro, ta, sp;

mat3 rotateMat(float roll, float pitch, float yaw)
{
    float cp = cos(pitch);
    float sp = sin(pitch);
    float sr = sin(roll);
    float cr = cos(roll);
    float sy = sin(yaw);
    float cy = cos(yaw);

    return mat3(cp * cy, (sr * sp * cy) - (cr * sy), (cr * sp * cy) + (sr * sy),
                cp * sy, (sr * sp * sy) + (cr * cy), (cr * sp * sy) - (sr * cy),
                -sp, sr * cp, cr * cp);
}

float stepUp(float t, float len, float smo)
{
    float tt = mod(t += smo, len);
    float stp = floor(t / len) - 1.0;
    return smoothstep(0.0, smo, tt) + stp;
}

float pingPong(float t, float len, float smo)
{
    t = mod(t + smo, len * 2.);
    return 1.0 - (smoothstep(0., smo, t) - smoothstep(len, len + smo, t));
}

float sphere( vec3 p, float s )
{
    return length(p)-s;
}

float sdBox( vec3 p, vec3 b )
{
    vec3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float de(vec3 p, mat3 rot, float scale) {
    vec3 offset = vec3(1,1,1);

    p*=transpose(rot);
    for (int i=0; i<Iterations; i++) {
        
        p*=rot;
        p = abs(p);

        if (p.x < p.y) {p.yx = mix(p.xy, p.yx, pingPong(beat, 63.5 * 0.5, 1.0));}
        if (p.x < p.z) {p.xz = mix(p.zx, p.xz, pingPong(beat, 63.5, 1.0));}
        if (p.y < p.z) {p.yz = mix(p.zy, p.yz, pingPong(beat, 63.5 * 0.25, 1.0));}

        p.z -= 0.5*offset.z*(scale-1.)/scale;
        p.z = -abs(-p.z);
        p.z += 0.5*offset.z*(scale-1.)/scale;

        p.xy = scale*p.xy - offset.xy*(scale-1.);
        p.z = scale*p.z;
    }

    vec3 d = abs(p) - vec3(1.,1.,1.);
    float distance = length(vec3(max(d, vec3(0.0))));
    distance *= pow(scale, -float(Iterations));
    
    return distance;
}

vec2 distStage(vec3 p, mat3 rot, float scale)
{
    float d = de(p, rot, scale);
    return vec2(d, MAT_STAGE);
}

vec2 distSphere(vec3 p)
{
    float wing = sphere(p, 0.1);
    float b1 = sdBox(p, vec3(10.0, 0.02, 10.0));
    float b2 = sdBox(p, vec3(0.02, 10.0, 10.0));
    float b3 = sdBox(p, vec3(10.0, 10.0, 0.02));
    float s = sphere(p, 0.098);
    wing = max(-b1, wing);
    wing = max(-b2, wing);
    wing = max(-b3, wing);
    wing = max(-s, wing);

    vec2 w = vec2(wing, MAT_WING);
    vec2 body = vec2(sphere(p, 0.08), MAT_BODY);
    return U(w, body);
}

vec2 distAll(vec3 p)
{
    vec3 pp = mod(p, 1.5) - 0.75;
    vec2 st1 = distStage(pp, stageRot, stageScale);
    vec2 st2 = distStage(pp, stageRot2 * stageRot, stageScale);
    vec2 sp = distSphere((p - sp) * sphereRot);
    return U(sp, U(st1, st2));
}

vec3 normal(vec3 pos, float e)
{
    vec3 eps = vec3(e,0.0,0.0);

    return normalize( vec3(
           distAll(pos+eps.xyy).x - distAll(pos-eps.xyy).x,
           distAll(pos+eps.yxy).x - distAll(pos-eps.yxy).x,
           distAll(pos+eps.yyx).x - distAll(pos-eps.yyx).x ) );
}

mat3 createCamera(vec3 ro, vec3 ta, float cr )
{
    vec3 cw = normalize(ta - ro);
    vec3 cp = vec3(sin(cr), cos(cr),0.0);
    vec3 cu = normalize( cross(cw,cp) );
    vec3 cv = normalize( cross(cu,cw) );
    return mat3( cu, cv, cw );
}

float softshadow( in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k)
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<4; i++ )
    {
        float h = distAll( ro + rd*t).x;
        res = min( res, k*h/t );
        t += clamp( h, 0.05, 0.2 );
        if( res<0.001 || t>maxt ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

mat2 rot(float x)
{
    return mat2(cos(x), sin(x), -sin(x), cos(x));
}

float sdRect( vec2 p, vec2 b )
{
    vec2 d = abs(p) - b;
    return min(max(d.x, d.y),0.0) + length(max(d,0.0));
}

float tex(vec2 p, float z)
{
    vec2 q = (fract(p / 10.0) - 0.5) * 10.0;
    float d = 9999.0;
    for (int i = 0; i < 3; ++i) {
        q = abs(q) - 0.5;
        q *= rot(0.785398);
        q = abs(q) - 0.5;
        q *= rot(z * 0.5);
        float k = sdRect(q, vec2(1.0, 0.55 + q.x));
        d = min(d, k);
    }
    float f = 1.0 / (1.0 + abs(d));
    return pow(f, 16.0) + smoothstep(0.9, 1.0, f);
}

vec3 light(vec3 pos, vec3 normal, vec3 ray, vec3 col, vec3 lpos, vec3 diffuse, vec3 specular, float smoothness)
{
    vec3 lvec = normalize(lpos - pos);
    vec3 hvec = normalize(lvec - ray);
    float llen = length(lpos - pos);
    float sha = (softshadow(pos, lvec, 0.01, length(lpos - pos), 4.0) + 0.25) / 1.25;
    vec3 diff = diffuse * col  * (1.0 / PI);

    float bpnorm = ( smoothness + 2.0 ) / ( 2.0 * PI );
    vec3 spec = specular * col * bpnorm * pow( max( 0.0, dot( normal, hvec ) ), smoothness );

    return vec3((diff + spec) * sha) / (llen * llen);
}

vec3 shade(vec3 pos, vec3 normal, vec3 ray, vec3 diffuse, vec3 specular, float smoothness)
{
    vec3 col = light(pos, normal, ray, vec3(0.01), ro, diffuse, specular, smoothness);
    col += light(pos, normal, ray, vec3(0.2, 0.4, 0.8), ro + vec3(0.0, 0.0, 2.0), diffuse, specular, smoothness);
    return col;
}

vec3 materialize(vec3 ro, vec3 ray, float depth, vec2 mat)
{
    vec3 pos = ro + ray * depth;
    vec3 nor = normal(pos, 0.0025);
    vec3 col = vec3(0.);

    if (mat.y == MAT_WING) {
        vec3 spLocalNormal = normalize((pos - sp) * sphereRot);
        vec3 pattern = 19.3602379925 * spLocalNormal;
        float emission = min(1.0,  tex(pattern.zy, 113.09) + tex(pattern.xz, 113.09) + tex(pattern.xy, 113.09));
        col += shade(pos, nor, ray, vec3(.25), vec3(.15), 10.);
        col += vec3(1.0, 0.25, 0.35) * 2. * emission * (sin(time) * 0.5 + 0.5 + 0.2);
    } else if (mat.y == MAT_BODY) {
        col += shade(pos, nor, ray, vec3(.25), vec3(.15), 5.);
        col += vec3(1.0, 0.25, 0.35) * 2. * (sin(time) * 0.5 + 0.5 + 0.2);
    } else if (mat.y == MAT_STAGE) {
        col += shade(pos, nor, ray, vec3(1.), vec3(1.), 25.);
    }

    return mix(col, vec3(0.1, 0.2, 0.4) * 80.0, pow(depth * 0.02, 2.1));
}

vec4 trace(vec3 ro, vec3 ray)
{
    float t = 0.0;
    vec2 res;
    for (int i = 0; i < 64; i++) {
        vec3 p = ro+ray*t;
        res = distAll(p);
        if( res.x < 0.0001 ) {
            break;
        }
        t += res.x;
    }
    return vec4(materialize(ro, ray, t, res), t);
}

void initTime(float t)
{
    time = t;
    beat = time * 120.0 / 60.0;
    //beat = mod(beat, 64.0);

    kick = mod(beat,1.);
    hihat = beat < 16.0 ? 0.0 : pingPong(beat + 0.5, 1.0, 0.1) * 0.1;
    snare = beat < 32.0 ? 0.0 : stepUp(beat - 32.5, 2.0, 0.5);
}

void scene1Init(vec2 p)
{
    stageScale = 3.4 - mix(0.00, 0.25, clamp(kick, 0.0, 1.0));
    stageRot = rotateMat(0.1-hihat,-hihat, 0.4-hihat);
    vec3 angle = mod(vec3(snare * 1.3, snare * 0.27, snare * 0.69), vec3(TAU) * 0.5);
    stageRot2 = rotateMat(angle.x, angle.y, angle.z);
    sphereRot = rotateMat(sin(time),cos(time), sin(time * .33));

    ro = (vec3(.75 + sin(time * 0.4) * 0.15, .8 + cos(time * 0.8) * 0.05, sin(time*0.3) * 0.05 + time * 0.5));
    ta = (vec3(0.75, 0.75,  (sin(time * 0.1) * 0.5 + 0.5) * 3.0 + 0.2 + time * 0.5));
    sp = (vec3(0.75, 0.75, 0.2 + time * 0.5));
    mat3 cm = createCamera(ro, ta, sin(time) * 0.1);
    ray = cm * normalize(vec3(p, 1.0));
}

vec3 scene(vec2 p)
{
    if (iTime < 6000.0) {
        initTime(iTime);
        scene1Init(p);
    }
    vec4 c = trace(ro, ray);
    return c.rgb;
}

vec2 hash( vec2 p ){
    p = vec2( dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3)));
    return fract(sin(p)*43758.5453) * 2.0 - 1.0;
}

float Bokeh(vec2 p, vec2 sp, float size, float mi, float blur)
{
    float d = length(p - sp);
    float c = smoothstep(size, size*(1.-blur), d);
    c *= mix(mi, 1., smoothstep(size*.8, size, d));
    return c;
}

vec3 dirt(vec2 uv, float n)
{
    vec2 p = fract(uv * n);
    vec2 st = (floor(uv * n) + 0.5) / n;
    vec2 rnd = hash(st);
    float c = Bokeh(p, vec2(0.5, 0.5) + vec2(0.3) * rnd, 0.2, abs(rnd.y * 0.4) + 0.3, 0.25 + rnd.x * rnd.y * 0.2);
    
    return vec3(c) * exp(rnd.x * 4.0);
}

vec3 postProcess(vec2 uv, vec3 col)
{   
    uv *= 0.5;
    
    vec3 di = dirt(uv, 3.5);
    di += dirt(uv - vec2(0.17), 3.0);
    di += dirt(uv- vec2(0.41), 2.75);
    di += dirt(uv- vec2(0.3), 2.5);

    float flare = pow(max(0.0, dot(vec3(0.0, 0.0, 1.0), ray)), 10.0);
    float flare2 = pow(max(0.0, dot(vec3(0.0, 0.0, 1.0), ray)), 8.0);
    vec3 f = flare * vec3(0.1, 0.2, 0.4) * 3.5 + flare2 * di * vec3(0.1, 0.2, 0.4) * 0.1;
    
    float sflare = pow(max(0.0, dot(normalize(sp - ro), ray)), 10.0);
    float sflare2 = pow(max(0.0, dot(normalize(sp - ro), ray)), 8.0);
    vec3 s = sflare * vec3(1.0, 0.25, 0.35) * 1.0 + sflare2 * di * vec3(1.0, 0.25, 0.35) * 0.01;
    
    return col + f + s * max(0.2, sin(time) * 0.5 + 0.5);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // fragment position
    vec2 p = (fragCoord.xy * 2.0 - iResolution.xy) / min(iResolution.x, iResolution.y);

    vec3 col =  scene(p);
    vec2 pp = fragCoord/iResolution.xy;
    col *= 0.5 + 0.5*pow( 16.0*pp.x*pp.y*(1.0-pp.x)*(1.0-pp.y), 0.05 );
    col = postProcess(p, col);
    fragColor = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}
