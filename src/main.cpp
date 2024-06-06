#include "logging.hpp"

#include <SDL.h>
#include <SDL_render.h>

#include <cglm/affine.h>
#include <cglm/cam.h>
#include <cglm/mat3.h>
#include <cglm/mat4.h>
#include <cglm/project.h>
#include <cglm/vec3.h>

#include <stdio.h>
#include <stdlib.h>

#define max( a, b )      ( a > b ? a : b )
#define min( a, b )      ( a < b ? a : b )
#define clamp( x, a, b ) ( x < a ? a : ( x > b ? b : x ) )

static unsigned int g_seed;

inline void fast_srand( int seed )
{
    g_seed = seed;
}

inline int fast_rand( void )
{
    g_seed = ( 214013 * g_seed + 2531011 );
    return ( g_seed >> 16 ) & 0x7FFF;
}

/// <returns> a random value in range [0.0f, 1.0f) </returns>
float rand_float()
{
    return (float) ( fast_rand() / ( RAND_MAX + 1.0 ) );
}

struct sphere_t {
    vec3 x;
    float r;
};

struct ray_t {
    vec3 x;
    vec3 dir;
};

struct color_t {
    float r;
    float g;
    float b;
};

static struct {
    SDL_Window * window;
    SDL_Renderer * renderer;
    SDL_Texture * texture;

    Uint8 * pixels;
    float * pixels_f;

    int width;
    int height;
    int should_close;

    float tick_step;
    Uint32 last_tick;

    int sample_count;
    int frame_num;
    float sample_weight;

    sphere_t * sphere_list;
    int sphere_count;

    sphere_t s;

    vec3 camera_pos;
    vec3 camera_dir;
    vec3 camera_right_dir;
    float camera_f;
    float camera_near;

    vec3 sun_dir;

    mat4 view;
    mat4 viewi;  // inverse
    mat3 view3i; // inverse
} intern;

static void random_scatter( vec3 out, vec3 normal )
{
resample:
    out[ 0 ] = rand_float() * 2.0f - 1.0f;
    out[ 1 ] = rand_float() * 2.0f - 1.0f;
    out[ 2 ] = rand_float() * 2.0f - 1.0f;

    if ( glm_vec3_norm2( out ) > 1.0f ) {
        goto resample;
    }

    glm_vec3_normalize( out );

    if ( glm_vec3_dot( out, normal ) < 0.0f ) {
        glm_vec3_negate( out );
    }
}

static void lambertian( vec3 out, vec3 normal )
{
resample:
    out[ 0 ] = rand_float() * 2.0f - 1.0f;
    out[ 1 ] = rand_float() * 2.0f - 1.0f;
    out[ 2 ] = rand_float() * 2.0f - 1.0f;

    if ( glm_vec3_norm2( out ) > 1.0f ) {
        goto resample;
    }

    glm_vec3_normalize( out );

    glm_vec3_add( out, normal, out );

    glm_vec3_normalize( out );
}

static void reflect( vec3 out, vec3 v, vec3 normal )
{
    float d = glm_vec3_dot( v, normal );
    vec3 z;
    glm_vec3_scale( normal, d * 2.0f, z );
    glm_vec3_sub( v, z, out );
}

static float intersect_sphere( sphere_t s, ray_t r )
{
    vec3 sx_rx; // sphere center to ray start
    glm_vec3_sub( r.x, s.x, sx_rx );

    // setup quadratic
    float a = glm_vec3_norm2( r.dir );
    float b = 2 * glm_vec3_dot( sx_rx, r.dir );
    float c = glm_vec3_norm2( sx_rx ) - s.r * s.r;

    // desciminant
    float det = b * b - 4 * a * c;

    // no hit
    if ( det <= 0 ) return 0.0f;

    // hit
    return ( -b - sqrtf( det ) ) / 2 * a;
}

static ray_t ray_from_pixel( float x, float y )
{
    ray_t r;

    float hw = intern.width * 0.5f;
    float hh = intern.height * 0.5f;

    r.dir[ 0 ] = ( x - hw ) * intern.camera_f;
    r.dir[ 1 ] = -( y - hh ) * intern.camera_f;
    r.dir[ 2 ] = -intern.camera_near;

    glm_vec3_normalize( r.dir );
    glm_mat3_mulv( intern.view3i, r.dir, r.dir );
    glm_vec3_copy( intern.camera_pos, r.x );

    return r;
}

static int hit_spheres( int * out_i, float * out_t, ray_t r )
{
    float depth = FLT_MAX;
    for ( int i = 0; i < intern.sphere_count; i++ ) {
        sphere_t s = intern.sphere_list[ i ];
        float t = intersect_sphere( s, r );

        if ( t <= 0.0f ) continue;
        if ( t > depth ) continue;

        *out_i = i;
        *out_t = t;

        depth = t;
    }

    return ( depth != FLT_MAX );
}

static color_t trace( ray_t r, int bounces_left )
{
    if ( bounces_left <= 0 ) {
        return color_t{ 0.0f, 0.0f, 0.0f };
    }

    int sphere_index;
    float hit_t;

    if ( hit_spheres( &sphere_index, &hit_t, r ) ) {
        sphere_t s = intern.sphere_list[ sphere_index ];

        vec3 hit;
        glm_vec3_copy( r.x, hit );
        glm_vec3_muladds( r.dir, hit_t, hit );

        vec3 s_normal;
        glm_vec3_sub( hit, s.x, s_normal );
        glm_vec3_scale( s_normal, 1.0f / s.r, s_normal );

        if ( sphere_index == 0 ) {
            ray_t incoming_r;
            reflect( incoming_r.dir, r.dir, s_normal );
            glm_vec3_copy( hit, incoming_r.x );
            // glm_vec3_muladds( incoming_r.dir, 0.001f, incoming_r.x );

            color_t incoming_color = trace( incoming_r, bounces_left - 1 );
            return incoming_color;
        } else {
            ray_t incoming_r;
            lambertian( incoming_r.dir, s_normal );
            glm_vec3_copy( hit, incoming_r.x );
            // glm_vec3_muladds( incoming_r.dir, 0.001f, incoming_r.x );

            color_t incoming_color = trace( incoming_r, bounces_left - 1 );
            incoming_color.r *= 0.8f;
            incoming_color.g *= 0.8f;
            incoming_color.b *= 0.8f;
            return incoming_color;
        }

    } else {
        // hit the sky

        // float light = glm_vec3_dot( r.dir, intern.sun_dir );
        // light = max( light, 0.0f );

        float light = 0.5f * ( r.dir[ 1 ] + 1.0f );

        color_t c;
        c.r = ( 1 - light ) * 1.0f + light * 0.8f;
        c.g = ( 1 - light ) * 1.0f + light * 0.0f;
        c.b = ( 1 - light ) * 1.0f + light * 0.0f;

        return c;
    }
}

static color_t trace_pixel( int x, int y )
{
    color_t c{ 0.0f, 0.0f, 0.0f };

    int sample_count = 1;
    float sample_w = 1.0f / sample_count;

    for ( int i = 0; i < sample_count; i++ ) {
        float dx = rand_float() - 0.5f;
        float dy = rand_float() - 0.5f;
        ray_t r = ray_from_pixel( x + dx, y + dy );

        color_t sample_c = trace( r, 4 );

        c.r += sample_w * sample_c.r;
        c.g += sample_w * sample_c.g;
        c.b += sample_w * sample_c.b;
    }

    return c;
}

static void reset_pixels()
{
    for ( int i = 0; i < intern.width * intern.height * 3; i++ ) {
        intern.pixels[ i ] = 0;
        intern.pixels_f[ i ] = 0;
    }
    intern.frame_num = 0;
}

static void setup_camera()
{
    vec3 up{ 0.0f, 1.0f, 0.0f };
    glm_look( intern.camera_pos, intern.camera_dir, up, intern.view );

    glm_mat4_inv( intern.view, intern.viewi );
    glm_mat4_pick3( intern.viewi, intern.view3i );
}

static void setup_spheres()
{
    intern.sphere_list = new sphere_t[ 32 ];
    intern.sphere_count = 3;

    intern.sphere_list[ 0 ] = intern.s;

    intern.sphere_list[ 1 ].x[ 0 ] = 0.0f;
    intern.sphere_list[ 1 ].x[ 1 ] = -105.0f;
    intern.sphere_list[ 1 ].x[ 2 ] = 10.0f;
    intern.sphere_list[ 1 ].r = 100.0f;

    intern.sphere_list[ 2 ].x[ 0 ] = 8.0f;
    intern.sphere_list[ 2 ].x[ 1 ] = -3.0f;
    intern.sphere_list[ 2 ].x[ 2 ] = 10.0f;
    intern.sphere_list[ 2 ].r = 2.0f;
}

static void setup()
{
    intern.width = 512;
    intern.height = 512;

    intern.sample_count = 1024;
    intern.sample_weight = 1.0f / intern.sample_count;

    // intern.width = 1024;
    // intern.height = 1024;

    {
        float hw = intern.width * 0.5f;
        float hh = intern.height * 0.5f;
        float near = 0.01f;
        float fovy = glm_rad( 45.0f );

        intern.camera_f = ( near * tanf( fovy / 2 ) ) / hh;
        intern.camera_near = near;
    }

    intern.pixels = new Uint8[ intern.width * intern.height * 3 ];   // 24-bit
    intern.pixels_f = new float[ intern.width * intern.height * 3 ]; // 24-bit

    intern.tick_step = 1 / 60.0f;

    // unit sphere at the center
    intern.s.x[ 0 ] = 0.0f;
    intern.s.x[ 1 ] = 0.0f;
    intern.s.x[ 2 ] = 15.0f;
    intern.s.r = 5.0f;

    intern.camera_pos[ 0 ] = 0.0f;
    intern.camera_pos[ 1 ] = 0.0f;
    intern.camera_pos[ 2 ] = 0.0f;

    glm_vec3_sub( intern.s.x, intern.camera_pos, intern.camera_dir );
    glm_vec3_normalize( intern.camera_dir );

    setup_camera();
    setup_spheres();

    reset_pixels();

    vec3 sun_dir{ 1.0f, -1.0f, 1.0f };
    glm_vec3_normalize( sun_dir );
    glm_vec3_negate( sun_dir );
    glm_vec3_copy( sun_dir, intern.sun_dir );

    ray_t r = ray_from_pixel( intern.width / 2, intern.height / 2 );
    INFO_LOG( "r = %f %f %f", r.dir[ 0 ], r.dir[ 1 ], r.dir[ 2 ] );
}

static void update_pixels()
{
#pragma omp parallel for
    for ( int x = 0; x < intern.width; x++ ) {
        for ( int y = 0; y < intern.height; y++ ) {
            int i = ( y * intern.width + x ) * 3;

            color_t c = trace_pixel( x, y );

            float w = intern.sample_weight;
            intern.pixels_f[ i + 0 ] += w * c.r;
            intern.pixels_f[ i + 1 ] += w * c.g;
            intern.pixels_f[ i + 2 ] += w * c.b;

            intern.pixels[ i + 0 ] =
                (Uint8) ( intern.pixels_f[ i + 0 ] * 255.0f );
            intern.pixels[ i + 1 ] =
                (Uint8) ( intern.pixels_f[ i + 1 ] * 255.0f );
            intern.pixels[ i + 2 ] =
                (Uint8) ( intern.pixels_f[ i + 2 ] * 255.0f );
        }
    }
}

static void render()
{
    setup_camera();

    if ( intern.frame_num < intern.sample_count ) {
        update_pixels();

        intern.frame_num++;

        SDL_UpdateTexture(
            intern.texture,
            nullptr,
            intern.pixels,
            intern.width * 3
        );
    }

    SDL_RenderClear( intern.renderer );
    SDL_RenderCopy( intern.renderer, intern.texture, nullptr, nullptr );
    SDL_RenderPresent( intern.renderer );
}

static void move_camera( int x, int y )
{
    float yaw = x * 0.001f;
    float pitch = -y * 0.001f;

    intern.camera_dir[ 0 ] = cos( pitch ) * cos( yaw );
    intern.camera_dir[ 1 ] = sin( pitch );
    intern.camera_dir[ 2 ] = cos( pitch ) * sin( yaw );

    vec3 up{ 0.0f, 1.0f, 0.0f };

    glm_vec3_cross( intern.camera_dir, up, intern.camera_right_dir );
}

static void tick()
{
    Uint32 current_tick = SDL_GetTicks();
    intern.tick_step = ( current_tick - intern.last_tick ) / 1000.0f;
    intern.last_tick = current_tick;

    static float timer = 1.0f;

    timer -= intern.tick_step;

    if ( timer <= 0.0f ) {
        char buffer[ 1024 ];

        snprintf(
            buffer,
            1024,
            "%d fps... meow meow meow meow meow",
            (int) ( 1.0f / intern.tick_step )
        );

        SDL_SetWindowTitle( intern.window, buffer );

        timer = 1.0f;
    }

    SDL_Event event;
    while ( SDL_PollEvent( &event ) ) {
        if ( event.type == SDL_QUIT ) intern.should_close = 1;
        if ( event.type == SDL_KEYDOWN ) {
            if ( event.key.keysym.scancode == SDL_SCANCODE_SPACE ) {
                reset_pixels();
            }
            if ( event.key.keysym.scancode == SDL_SCANCODE_ESCAPE ) {
                intern.should_close = 1;
            }
        }

        if ( event.type == SDL_MOUSEMOTION ) {
            static int x = 0;
            static int y = 0;

            x += event.motion.xrel;
            y += event.motion.yrel;

            move_camera( x, y );
        }
    }

    const Uint8 * keys = SDL_GetKeyboardState( nullptr );

    float forward = 0.0f;
    float right = 0.0f;
    if ( keys[ SDL_SCANCODE_W ] ) {
        forward = 1.0f;
    }
    if ( keys[ SDL_SCANCODE_A ] ) {
        right = -1.0f;
    }
    if ( keys[ SDL_SCANCODE_S ] ) {
        forward = -1.0f;
    }
    if ( keys[ SDL_SCANCODE_D ] ) {
        right = 1.0f;
    }

    glm_vec3_muladds(
        intern.camera_dir,
        forward * 10.0f * intern.tick_step,
        intern.camera_pos
    );

    glm_vec3_muladds(
        intern.camera_right_dir,
        right * 10.0f * intern.tick_step,
        intern.camera_pos
    );
}

int main( int argc, char ** argv )
{
    setup();

    INFO_LOG( "meow" );

    SDL_Init( SDL_INIT_VIDEO );
    SDL_CreateWindowAndRenderer(
        intern.width,
        intern.height,
        0,
        &intern.window,
        &intern.renderer
    );
    SDL_SetWindowTitle(
        intern.window,
        "meow meow meow meow meow meow meow meow"
    );

    SDL_SetRelativeMouseMode( SDL_TRUE );

    intern.texture = SDL_CreateTexture(
        intern.renderer,
        SDL_PIXELFORMAT_RGB24,
        SDL_TEXTUREACCESS_STATIC,
        intern.width,
        intern.height
    );

    while ( !intern.should_close ) {
        tick();
        render();
    }

    SDL_DestroyRenderer( intern.renderer );
    SDL_DestroyWindow( intern.window );
    SDL_Quit();

    return 0;
}