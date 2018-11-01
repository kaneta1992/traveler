#define TAU 6.283185307
#define PI 3.141592654
#define U(z,w) (z.x < w.x ? z : w)

#define MAT_WING  1.0
#define MAT_BODY  2.0
#define MAT_STAGE 3.0
#define saturate(x) (clamp(x, 0.0, 1.0))

const int Iterations = 3;

float time;
float beat, sceneBeat, kick, hihat, snare;
float stageScale;
float edgeOnly;
vec3 fogColor;
mat3 sphereRot, stageRot, stageRot2;
vec3 ray;
vec3 ro, ta, sp;
vec3 cameraLight, stageLight;
vec3 stageFlareCol, travelerFlareCol;
float stageFlareIntensity, travelerFlareIntensity, stageFlareExp, travelerFlareExp;

float sm(float start, float end, float t, float smo)
{
    return smoothstep(start, start + smo, t) - smoothstep(end - smo, end, t);
}

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

float glowTime(vec3 p)
{
    float t = beat - 48.0;
    if (t < 0.0) {
        return -1.0;
    }
    t -= 1.0;
    float len = distance(sp, p);
    float tt = mod(t, 8.) + floor(len / 8.0) * 8.0;
    return tt;
}

float patternIntensity(vec3 p)
{
    float t = beat - 32.0;
    if (t < 0.0) {
        return 0.0;
    }
    t -= 2.0;
    float len = distance(sp, p);
    return sm(0.0, 2., mod(len - t * 1.5, 6.0), .5);
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

        if (beat < 48.0) {
            if (p.x < p.y) {p.yx = p.xy;}
            if (p.x < p.z) {p.xz = p.zx;}
            if (p.y < p.z) {p.yz = p.zy;}
        } else {
            if (p.x < p.y) {p.yx = mix(p.xy, p.yx, pingPong(sceneBeat, 63.5 * 0.5, 1.0));}
            if (p.x < p.z) {p.xz = mix(p.zx, p.xz, pingPong(sceneBeat, 63.5, 1.0));}
            if (p.y < p.z) {p.yz = mix(p.zy, p.yz, pingPong(sceneBeat, 63.5 * 0.25, 1.0));}
        }

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

vec2 distGlow(vec3 p)
{
    vec3 pp = mod(p, 1.5) - 0.75;
    vec2 st1 = distStage(pp, stageRot, stageScale);
    vec2 st2 = distStage(pp, stageRot2 * stageRot, stageScale);

    float gt = glowTime(p);

    float frontSp = sphere(p - sp, gt + 1.);
    float backSp = sphere(p - sp, gt);
    float cut = max(frontSp, -backSp);
    vec2 st = U(st1, st2);
    st.x = max(st.x, cut);
    return st;
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

float tex2(vec2 p, float z)
{
    vec2 q = (fract(p / 10.0) - 0.5) * 10.0;
    float d = 9999.0;
    for (int i = 0; i < 2; ++i) {
        q = abs(q) - 0.5;
        q *= rot(0.785398);
        q = abs(q) - 0.5;
        q *= rot(z * 0.5);
        float k = sdRect(q, vec2(1.0, 0.55 + q.x));
        d = min(d, k);
    }
    float f = 1.0 / (1.0 + abs(d));
    return pow(f, 16.0) + smoothstep(0.95, 1.0, f);
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
    return pow(f, 16.0) + smoothstep(0.95, 1.0, f);
}

vec3 light(vec3 pos, vec3 normal, vec3 ray, vec3 col, vec3 lpos, vec3 diffuse, vec3 specular, float smoothness)
{
    vec3 lvec = normalize(lpos - pos);
    vec3 hvec = normalize(lvec - ray);
    float llen = length(lpos - pos);
    vec3 diff = diffuse * col  * (1.0 / PI);

    float bpnorm = ( smoothness + 2.0 ) / ( 2.0 * PI );
    vec3 spec = specular * col * bpnorm * pow( max( 0.0, dot( normal, hvec ) ), smoothness );

    return vec3(diff + spec) / (llen * llen);
}

vec3 shade(vec3 pos, vec3 normal, vec3 ray, vec3 diffuse, vec3 specular, float smoothness)
{
    vec3 col = light(pos, normal, ray, cameraLight, ro, diffuse, specular, smoothness);
    col += light(pos, normal, ray, stageLight, ro + vec3(0.0, 0.0, 2.0), diffuse, specular, smoothness);
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
        col += vec3(1.0, 0.25, 0.35) * 2. * emission * (cos(beat * 0.5) * 0.5 + 0.5 + 0.2);
    } else if (mat.y == MAT_BODY) {
        col += shade(pos, nor, ray, vec3(.25), vec3(.15), 5.);
        col += vec3(1.0, 0.25, 0.35) * 2. * (cos(beat * 0.5) * 0.5 + 0.5 + 0.2);
    } else if (mat.y == MAT_STAGE) {
        vec3 n = pos * 9.3602379925;
        float edge = tex2(n.zy, 113.09) + tex2(n.xz, 113.09) + tex2(n.xy, 113.09);
        vec3 lpos = ro + vec3(0.0, 0.0, 2.0);
        vec3 lvec = normalize(lpos - pos);
        float sha = (softshadow(pos, lvec, 0.01, length(lpos - pos), 4.0) + 0.25) / 1.25;

        // ステージが出現する演出
        float noShade = 0.0;
        if (beat > 16.0) {
            noShade = step(distance(pos, sp), sceneBeat);
        }

        col += shade(pos, nor, ray, vec3(1.), vec3(1.), 25.) *sha * edgeOnly * noShade + max(edge, 0.0) * vec3(0.1,0.2,0.4) * 4.0 * patternIntensity(pos);
    }

    return mix(col, fogColor, pow(depth * 0.02, 2.1));
}

vec3 rgb2hsv(vec3 hsv)
{
	vec4 t = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	vec3 p = abs(fract(vec3(hsv.x) + t.xyz) * 6.0 - vec3(t.w));
	return hsv.z * mix(vec3(t.x), clamp(p - vec3(t.x), 0.0, 1.0), hsv.y);
}

vec3 hash3( vec3 p ){
    vec3 q = vec3(dot(p,vec3(127.1,311.7, 114.5)), dot(p,vec3(269.5,183.3, 191.9)), dot(p,vec3(419.2,371.9, 514.1)));
    return fract(sin(q)*43758.5453);
}

vec3 glowTrace(vec3 ro, vec3 ray, float maxDepth)
{
    float t = 0.0;
    vec3 col = vec3(0.);
    for (int i = 0; i < 16; i++) {
        vec3 p = ro+ray*t;
        float len = distance(sp, p);
        float gt = glowTime(p);

        // 光らせたくないときは-1.0を返してる
        if (gt < 0.0) {
            return vec3(0.);
        }

        vec3 h = hash3(floor(p * 30.0) / 30.0) * 2.0 - 1.0;
        float val = 1.0 - sm(gt, gt + 2.0, len, .25);
        // TODO: smでバラバラ感を制御しているが思った挙動じゃないので調査する
        vec2 res = distGlow(p + h * 0.15 * val);
        col += max(vec3(0.0), 0.002 / res.x) * rgb2hsv(vec3(p.x * 1., 0.8, 1.0));
        t += res.x;
        if (maxDepth < t) {
            break;
        }
    }
    return col;
}

vec4 trace(vec3 ro, vec3 ray)
{
    float t = 0.0;
    float stepIntensity = 0.0;
    vec2 res;
    for (int i = 0; i < 64; i++) {
        vec3 p = ro+ray*t;
        res = distAll(p);
        if( res.x < 0.0001 ) {
            stepIntensity = float(i) / 64.0;
            break;
        }
        t += res.x;
    }
    vec3 p = ro + ray * t;
    float val = patternIntensity(p);
    vec3 sg1 = pow(stepIntensity * 1.0, 5.0) * vec3(.2, .4, .8) * val * 20.0;
    //vec3 sg2 = pow(stepIntensity * 1.0, 2.0) * vec3(1., 0., 0.) * ((sin(sceneBeat) + 1.0) * 0.5);
    return vec4(materialize(ro, ray, t, res) + sg1, t);
}

void initBeat(float b)
{
    sceneBeat = b;

    kick = mod(sceneBeat, 1.);
    hihat = sceneBeat < 16.0 ? 0.0 : pingPong(sceneBeat + 0.5, 1.0, 0.1) * 0.1;
    snare = sceneBeat < 32.0 ? 0.0 : stepUp(sceneBeat - 32.5, 2.0, 0.5);
}

void stageInit()
{
    stageScale = 3.4 - mix(0.00, 0.25, clamp(kick, 0.0, 1.0));
    stageRot = rotateMat(0.1-hihat,-hihat, 0.4-hihat);
    vec3 angle = mod(vec3(snare * 1.3, snare * 0.27, snare * 0.69), vec3(TAU) * 0.5);
    stageRot2 = rotateMat(angle.x, angle.y, angle.z);
    sphereRot = rotateMat(sin(time),cos(time), sin(time * .33));
}

void travelerInit(vec3 p)
{
    sp = p;
}

void cameraInit(vec2 p, vec3 origin, vec3 target, float angle, float fov)
{
    ro = origin;
    ta = target;
    mat3 cm = createCamera(ro, ta, angle);
    ray = cm * normalize(vec3(p, fov));
}

void fogInit(vec3 col)
{
    fogColor = col;
}

void stageEdgeOnly(float val)
{
    edgeOnly = 1.0 - val;
}

void initLight(vec3 camera, vec3 stage)
{
    cameraLight = camera;
    stageLight = stage;
}

void initFlare(vec3 stage, float stageIntensity, float stageExp, vec3 traveler, float travelerIntensity, float travelerExp)
{
    stageFlareCol = stage;
    stageFlareIntensity = stageIntensity;
    stageFlareExp = stageExp;
    travelerFlareCol = traveler;
    travelerFlareIntensity = travelerIntensity;
    travelerFlareExp = travelerExp;
}

vec2 hash( vec2 p ){
    p = vec2( dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3)));
    return fract(sin(p)*43758.5453) * 2.0 - 1.0;
}

float quadraticInOut(float t) {
  float p = 2.0 * t * t;
  return t < 0.5 ? p : -p + (4.0 * t) - 1.0;
}

vec3 scene(vec2 p)
{
    time = iTime;
    beat = time * 120.0 / 60.0;


    float val = sin(beat * 0.25);
    float scene0Beat = beat;
    float scene1Beat = beat - 16.;
    float scene2Beat = beat - 48.;
    vec3 scene1CameraPos = vec3(sin(scene1Beat * 0.5) * 0.3 + val * 0.05, .15 + val * 0.05, cos(scene1Beat * 0.5) * 0.3 + val * 0.05);
    vec3 scene2CameraPos = vec3(sin(scene2Beat * 0.2) * 0.15, cos(scene2Beat * 0.4) * 0.05 + 0.05, cos(scene2Beat * 0.15 + PI) * 0.05 - 0.2);

    vec3 scene1CameraTarget = vec3(0.);
    vec3 scene2CameraTarget = vec3(0.0, 0.0, (sin(scene2Beat * 0.05) * 0.5 + 0.5) * 3.0);

    if (beat < 16.0) {
        initBeat(scene0Beat);
        fogInit(vec3(0.0));
        stageEdgeOnly(1.0);
        travelerInit(vec3(0.75, 0.75, mix(-20.0, 20.0, sceneBeat / 16.0)));
        vec3 cameraPos = vec3(0.9, 0.8, 0.0);
        vec2 rnd = hash(vec2(sceneBeat * 0.5)) * 0.05 * max(0.0, 1.0 - distance(cameraPos, sp) / 5.0);
        cameraPos.xy += rnd;
        cameraInit(p, cameraPos,
                    vec3(vec2(0.75, 0.75) + rnd, 1.0),
                    0.0,
                    3.0);
        initLight(vec3(0.01), vec3(0.0));
        initFlare(vec3(0.2, 0.4, 0.8) * 1.5, 0.0, 1.0, vec3(1.0, 0.25, 0.35), max(0.2, cos(sceneBeat * 0.5) * 0.5 + 0.5),  mix(1.0, 800.0, distance(ro, sp) / 10.0));
    } else if (beat < 48.0) {
        initBeat(scene1Beat);
        fogInit(vec3(0.0));
        stageEdgeOnly(1.0);
        travelerInit(vec3(0.75, 0.75, 0.2 + beat * 0.25));
        float animVal = saturate(beat - 47.0);
        cameraInit(p, mix(sp + scene1CameraPos, sp + scene2CameraPos, quadraticInOut(animVal)),
                    mix(sp + scene1CameraTarget, sp + scene2CameraTarget, quadraticInOut(animVal * animVal)),
                    mix(val * 0.1, sin(scene2Beat * 0.5) * 0.1, quadraticInOut(animVal)),
                    mix(2.5, 1., quadraticInOut(animVal * animVal)));
        initLight(vec3(0.01), vec3(0.0));
        initFlare(vec3(0.2, 0.4, 0.8) * 1.5, 0.0, 1.0, vec3(1.0, 0.25, 0.35), max(0.2, cos(beat * 0.5) * 0.5 + 0.5), 8.0);
    } else if (beat < 6000.0) {
        initBeat(scene2Beat);
        fogInit(mix(vec3(0.0), vec3(0.1, 0.2, 0.4) * 80.0, vec3(saturate((sceneBeat - 2.0) * 0.5))));
        stageEdgeOnly(0.0);
        travelerInit(vec3(0.75, 0.75, 0.2 + beat * 0.25));
        cameraInit(p, sp + scene2CameraPos,
                    sp + scene2CameraTarget,
                    sin(sceneBeat * 0.5) * 0.1,
                    1.0);
        initLight(vec3(0.01), vec3(0.2, 0.4, 0.8));
        initFlare(vec3(0.2, 0.4, 0.8) * 1.5, mix(0.0, 1.0, saturate((sceneBeat - 2.0) * 0.5)), 8.0, vec3(1.0, 0.25, 0.35), max(0.2, cos(beat * 0.5) * 0.5 + 0.5), 8.0);
    }
    stageInit();
    vec4 c = trace(ro, ray);
    c.rgb += glowTrace(ro, ray, c.w + 0.01);
    return c.rgb;
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
    di += dirt(uv - vec2(0.47), 3.5);
    di += dirt(uv- vec2(0.21), 4.0);
    di += dirt(uv- vec2(0.6), 4.5);

    float flare = pow(max(0.0, dot(vec3(0.0, 0.0, 1.0), ray)), stageFlareExp * 1.25);
    float flare2 = pow(max(0.0, dot(vec3(0.0, 0.0, 1.0), ray)), stageFlareExp);
    vec3 f = flare * stageFlareCol + flare2 * di * stageFlareCol * 0.05;
    
    float sflare = pow(max(0.0, dot(normalize(sp - ro), ray)), travelerFlareExp * 1.25);
    float sflare2 = pow(max(0.0, dot(normalize(sp - ro), ray)), travelerFlareExp);
    vec3 s = sflare * travelerFlareCol + sflare2 * di * travelerFlareCol * 0.05;
    
    return col + f * stageFlareIntensity + s * travelerFlareIntensity;
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
