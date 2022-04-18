/* 
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//-----------------------------------------------------------------------------
//
// optixMeshViewer: simple interactive mesh viewer 
//
//-----------------------------------------------------------------------------

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#    include <GL/freeglut.h>
#  else
#    include <GL/glut.h>
#  endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <sutil.h>
#include "common.h"
#include <Arcball.h>
#include <OptiXMesh.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdint.h>

using namespace optix;

const char* const SAMPLE_NAME = "optixMeshViewer";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

optix::Context        context;
uint32_t       width  = 1024u;
uint32_t       height = 768u;
bool           use_pbo = true;
bool           use_tri_api = true;
bool           ignore_mats = false;
optix::Aabb    aabb;

// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
sutil::Arcball arcball;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;


//------------------------------------------------------------------------------
//
// Forward decls 
//
//------------------------------------------------------------------------------

struct UsageReportLogger;

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext( int usage_report_level, UsageReportLogger* logger );
void loadMesh( const std::string& filename );
void setupCamera();
void setupLights();
void updateCamera();
void glutInitialize( int* argc, char** argv );
void glutRun();

void glutDisplay();
void glutKeyboardPress( unsigned char k, int x, int y );
void glutMousePress( int button, int state, int x, int y );
void glutMouseMotion( int x, int y);
void glutResize( int w, int h );


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer()
{
    return context[ "output_buffer" ]->getBuffer();
}


void destroyContext()
{
    if( context )
    {
        context->destroy();
        context = 0;
    }
}


struct UsageReportLogger
{
  void log( int lvl, const char* tag, const char* msg )
  {
    std::cout << "[" << lvl << "][" << std::left << std::setw( 12 ) << tag << "] " << msg;
  }
};

// Static callback
void usageReportCallback( int lvl, const char* tag, const char* msg, void* cbdata )
{
    // Route messages to a C++ object (the "logger"), as a real app might do.
    // We could have printed them directly in this simple case.

    UsageReportLogger* logger = reinterpret_cast<UsageReportLogger*>( cbdata );
    logger->log( lvl, tag, msg ); 
}

void registerExitHandler()
{
    // register shutdown handler
#ifdef _WIN32
    glutCloseFunc( destroyContext );  // this function is freeglut-only
#else
    atexit( destroyContext );
#endif
}


void createContext( int usage_report_level, UsageReportLogger* logger )
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );
	context->setStackSize(2800);
	context->setMaxTraceDepth(30);
	context["max_depth"]->setInt(20);
	context["importance_cutoff"]->setFloat(0.01f);
    if( usage_report_level > 0 )
    {
        context->setUsageReportCallback( usageReportCallback, usage_report_level, logger );
    }

    context["scene_epsilon"    ]->setFloat( 1.e-4f );

    Buffer buffer = sutil::createOutputBuffer( context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo );
    context["output_buffer"]->set( buffer );

    // Ray generation program
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "pinhole_camera.cu" );
    Program ray_gen_program = context->createProgramFromPTXString( ptx, "pinhole_camera" );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXString( ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    context->setMissProgram( 0, context->createProgramFromPTXString( sutil::getPtxString( SAMPLE_NAME, "constantbg.cu" ), "miss" ) );
    context["bg_color"]->setFloat( 0.34f, 0.55f, 0.85f );
}


void loadMesh( const std::string& filename )
{
	//测试了一些位移与旋转
	std::string mesh_file = std::string("D:/model/bun_zipper_res2.ply");
    OptiXMesh mesh;
    mesh.context = context;
    mesh.use_tri_api = use_tri_api;
    mesh.ignore_mats = ignore_mats;
   

	


	std::string mesh_file2 = std::string("D:/model/dragon.ply");
	//std::string mesh_file2 = std::string("D:/model/bunny.ply");
	const char *ptx;
	OptiXMesh mesh2;
	mesh2.context = context;
	//mesh2.use_tri_api = use_tri_api;
	//mesh2.ignore_mats = ignore_mats;
	
	// Glass material
	Material glass_matl;
	ptx = sutil::getPtxString(SAMPLE_NAME, "glass.cu");
	Program glass_ch = context->createProgramFromPTXString(ptx, "glass_closest_hit_radiance");
	Program glass_ah = context->createProgramFromPTXString(ptx, "glass_any_hit_shadow");
	glass_matl = context->createMaterial();
	glass_matl->setClosestHitProgram(0, glass_ch);
	glass_matl->setAnyHitProgram(1, glass_ah);

	glass_matl["importance_cutoff"]->setFloat(1e-2f);
	glass_matl["cutoff_color"]->setFloat(0.34f, 0.55f, 0.85f);
	glass_matl["fresnel_exponent"]->setFloat(3.0f);
	glass_matl["fresnel_minimum"]->setFloat(0.1f);
	glass_matl["fresnel_maximum"]->setFloat(1.0f);
	glass_matl["refraction_index"]->setFloat(1.4f);
	glass_matl["refraction_color"]->setFloat(1.0f, 1.0f, 1.0f);
	glass_matl["reflection_color"]->setFloat(1.0f, 1.0f, 1.0f);
	glass_matl["refraction_maxdepth"]->setInt(100);
	glass_matl["reflection_maxdepth"]->setInt(100);
	float3 extinction = make_float3(.80f, .89f, .75f);
	glass_matl["extinction_constant"]->setFloat(log(extinction.x), log(extinction.y), log(extinction.z));
	glass_matl["shadow_attenuation"]->setFloat(0.4f, 0.7f, 0.4f);
	mesh.material = glass_matl;
	mesh.closest_hit = glass_ch;
	mesh.any_hit = glass_ah;
	//mesh2.material = glass_matl;
	//mesh2.closest_hit = glass_ch;
	//mesh2.any_hit = glass_ah;

	loadMesh(mesh_file, mesh);
	loadMesh(mesh_file2, mesh2);




	GeometryGroup geometry_group = context->createGeometryGroup();
	geometry_group->addChild(mesh.geom_instance);
	geometry_group->setAcceleration(context->createAcceleration("Trbvh"));

	GeometryGroup geometry_group2 = context->createGeometryGroup();
	geometry_group2->addChild(mesh2.geom_instance);
	geometry_group2->setAcceleration(context->createAcceleration("Trbvh"));

    aabb.set( mesh.bbox_min, mesh.bbox_max );


	Transform transform = context->createTransform();
	Transform transform2 = context->createTransform();
	Transform transform3 = context->createTransform();
	Transform transform4 = context->createTransform();
	Transform transform5 = context->createTransform();
	/*
	float m[16] = {
		1.0f, 0.0f, 0.0f, 1.0 * 3.0f - 1.5f,
		0.0f, 1.0f, 0.0f, 0.5f,
		0.0f, 0.0f, 1.0f, 1.0 * 2.5f - 1.5f,
		0.0f, 0.0f, 0.0f, 1.0f
	};
	*/
	float max_dim = fmaxf(aabb.extent(0), aabb.extent(1));
	
	float m[16] = {
		1.0f, 0.0f, 0.0f, -max_dim*0.01,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0,
		0.0f, 0.0f, 0.0f, 1.0f
	};
	
	float m2[16] = {
	1.0f, 0.0f, 0.0f, 0.0,
	0.0f, 1.0f, 0.0f,-max_dim*0.25f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
	};
	float m3[16] = {
	0.5f, 0.0f, 0.0f, 0.0,
	0.0f, 0.5f, 0.0f, 0.0f,
	0.0f, 0.0f, 0.5f, 0.0,
	0.0f, 0.0f, 0.0f, 0.5f
	};
	float m4[16] = {
	1.0f, 0.0f, 0.0f, max_dim*1.4,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, -max_dim * 0.6,
	0.0f, 0.0f, 0.0f, 1.0f
	};
	float theta = 3.15159/2.0/2.0;
	 theta =0.0f;
	/*
	float rotate[16] = {
	cos(theta), -sin(theta), 0.0f, 0.0f,
	sin(theta), cos(theta), 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0,
	0.0f, 0.0f, 0.0f, 1.0f
	};
	*/
	float rotate[16] = {
	cos(theta), 0.0f, sin(theta), 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	-sin(theta), 0.0f, cos(theta), 0.0,
	0.0f, 0.0f, 0.0f, 1.0f
	};
	transform->setMatrix(false, rotate, NULL);
	transform2->setMatrix(false, m, NULL);
	transform3->setMatrix(false, m3, NULL);
	transform4->setMatrix(false, m4, NULL);
	transform5->setMatrix(false, m2, NULL);
	transform->setChild(geometry_group2);
	transform2->setChild(transform);
	transform3->setChild(transform2);
	transform4->setChild(geometry_group);
	transform5->setChild(transform2);


	// floor geo
	Geometry parallelogram = context->createGeometry();
	parallelogram->setPrimitiveCount(1u);
	ptx = sutil::getPtxString(SAMPLE_NAME, "parallelogram.cu");
	parallelogram->setBoundingBoxProgram(context->createProgramFromPTXString(ptx, "bounds"));
	parallelogram->setIntersectionProgram(context->createProgramFromPTXString(ptx, "intersect"));
	float3 anchor = make_float3(-1.0f, 0.01f, -1.0f);
	float3 v1 = make_float3(2.0f, 0.0f, 0.0f);
	float3 v2 = make_float3(0.0f, 0.0f, 2.0f);
	float3 normal = cross(v1, v2);
	normal = normalize(normal);
	float d = dot(normal, anchor);
	v1 *= 1.0f / dot(v1, v1);
	v2 *= 1.0f / dot(v2, v2);
	float4 plane = make_float4(normal, d);
	parallelogram["plane"]->setFloat(plane);
	parallelogram["v1"]->setFloat(v1);
	parallelogram["v2"]->setFloat(v2);
	parallelogram["anchor"]->setFloat(anchor);


	// checker material
	ptx = sutil::getPtxString(SAMPLE_NAME, "checker.cu");
	Program check_ch = context->createProgramFromPTXString(ptx, "closest_hit_radiance");
	Program check_ah = context->createProgramFromPTXString(ptx, "any_hit_shadow");
	Material floor_matl = context->createMaterial();
	floor_matl->setClosestHitProgram(0, check_ch);
	floor_matl->setAnyHitProgram(1, check_ah);

	floor_matl["Kd1"]->setFloat(0.8f, 0.3f, 0.15f);
	floor_matl["Ka1"]->setFloat(0.8f, 0.3f, 0.15f);
	floor_matl["Ks1"]->setFloat(0.0f, 0.0f, 0.0f);
	floor_matl["Kd2"]->setFloat(0.9f, 0.85f, 0.05f);
	floor_matl["Ka2"]->setFloat(0.9f, 0.85f, 0.05f);
	floor_matl["Ks2"]->setFloat(0.0f, 0.0f, 0.0f);
	floor_matl["inv_checker_size"]->setFloat(10.0f, 10.0f, 1.0f);
	floor_matl["phong_exp1"]->setFloat(0.0f);
	floor_matl["phong_exp2"]->setFloat(0.0f);
	floor_matl["Kr1"]->setFloat(0.0f, 0.0f, 0.0f);
	floor_matl["Kr2"]->setFloat(0.0f, 0.0f, 0.0f);

	


	std::vector<GeometryInstance> gis;
	gis.push_back(context->createGeometryInstance(parallelogram, &floor_matl, &floor_matl + 1));
	GeometryGroup floorgeo = context->createGeometryGroup();
	floorgeo->setChildCount(static_cast<unsigned int>(gis.size()));
	floorgeo->setChild(0, gis[0]);
	floorgeo->setAcceleration(context->createAcceleration("Trbvh"));



	Group top_group = context->createGroup();
	top_group->setAcceleration(context->createAcceleration( "Trbvh"));
	top_group->setChildCount(3);

	top_group->setChild(0, transform5);
	top_group->setChild(1, transform4);
	top_group->setChild(2, floorgeo);
    context[ "top_object"   ]->set(top_group);
    context[ "top_shadower" ]->set(top_group);
}

  
void setupCamera()
{
    const float max_dim = fmaxf(aabb.extent(0), aabb.extent(1)); // max of x, y components

    camera_eye    = aabb.center() + make_float3( 0.0f, 0.0f, max_dim*3.0f ); 
    camera_lookat = aabb.center(); 
    camera_up     = make_float3( 0.0f, 1.0f, 0.0f );

    camera_rotate  = Matrix4x4::identity();
}


void setupLights()
{
    const float max_dim = fmaxf(aabb.extent(0), aabb.extent(1)); // max of x, y components

    BasicLight lights[] = {
        { make_float3( -0.5f,  0.25f, -1.0f ), make_float3( 0.2f, 0.2f, 0.25f ), 0, 0 },
        { make_float3( -0.5f,  0.0f ,  1.0f ), make_float3( 0.1f, 0.1f, 0.10f ), 0, 0 },
        { make_float3(  0.5f,  0.5f ,  0.5f ), make_float3( 0.7f, 0.7f, 0.65f ), 1, 0 }
    };
    lights[0].pos *= max_dim * 10.0f; 
    lights[1].pos *= max_dim * 10.0f; 
    lights[2].pos *= max_dim * 10.0f; 

    Buffer light_buffer = context->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( BasicLight ) );
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    context[ "lights" ]->set( light_buffer );
}


void updateCamera()
{
    const float vfov = 35.0f;
    const float aspect_ratio = static_cast<float>(width) /
                               static_cast<float>(height);
    
    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    const Matrix4x4 frame = Matrix4x4::fromBasis( 
            normalize( camera_u ),
            normalize( camera_v ),
            normalize( -camera_w ),
            camera_lookat);
    const Matrix4x4 frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const Matrix4x4 trans     = frame*camera_rotate*camera_rotate*frame_inv; 

    camera_eye    = make_float3( trans*make_float4( camera_eye,    1.0f ) );
    camera_lookat = make_float3( trans*make_float4( camera_lookat, 1.0f ) );
    camera_up     = make_float3( trans*make_float4( camera_up,     0.0f ) );

    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    camera_rotate = Matrix4x4::identity();

    context["eye"]->setFloat( camera_eye );
    context["U"  ]->setFloat( camera_u );
    context["V"  ]->setFloat( camera_v );
    context["W"  ]->setFloat( camera_w );
}


void glutInitialize( int* argc, char** argv )
{
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 100, 100 );                                               
    glutCreateWindow( SAMPLE_NAME );
    glutHideWindow();                                                              
}


void glutRun()
{
    // Initialize GL state                                                            
    glMatrixMode(GL_PROJECTION);                                                   
    glLoadIdentity();                                                              
    glOrtho(0, 1, 0, 1, -1, 1 );                                                   

    glMatrixMode(GL_MODELVIEW);                                                    
    glLoadIdentity();                                                              

    glViewport(0, 0, width, height);                                 

    glutShowWindow();                                                              
    glutReshapeWindow( width, height);

    // register glut callbacks
    glutDisplayFunc( glutDisplay );
    glutIdleFunc( glutDisplay );
    glutReshapeFunc( glutResize );
    glutKeyboardFunc( glutKeyboardPress );
    glutMouseFunc( glutMousePress );
    glutMotionFunc( glutMouseMotion );

    registerExitHandler();

    glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
    updateCamera();
    context->launch( 0, width, height );

    sutil::displayBufferGL( getOutputBuffer() );

    {
      static unsigned frame_count = 0;
      sutil::displayFps( frame_count++ );
    }

    glutSwapBuffers();
}


void glutKeyboardPress( unsigned char k, int x, int y )
{

    switch( k )
    {
        case( 'q' ):
        case( 27 ): // ESC
        {
            destroyContext();
            exit(0);
        }
        case( 's' ):
        {
            const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
            std::cerr << "Saving current frame to '" << outputImage << "'\n";
            sutil::displayBufferPPM( outputImage.c_str(), getOutputBuffer() );
            break;
        }
    }
}


void glutMousePress( int button, int state, int x, int y )
{
    if( state == GLUT_DOWN )
    {
        mouse_button = button;
        mouse_prev_pos = make_int2( x, y );
    }
    else
    {
        // nothing
    }
}


void glutMouseMotion( int x, int y)
{
    if( mouse_button == GLUT_RIGHT_BUTTON )
    {
        const float dx = static_cast<float>( x - mouse_prev_pos.x ) /
                         static_cast<float>( width );
        const float dy = static_cast<float>( y - mouse_prev_pos.y ) /
                         static_cast<float>( height );
        const float dmax = fabsf( dx ) > fabs( dy ) ? dx : dy;
        const float scale = fminf( dmax, 0.9f );
        camera_eye = camera_eye + (camera_lookat - camera_eye)*scale;
    }
    else if( mouse_button == GLUT_LEFT_BUTTON )
    {
        const float2 from = { static_cast<float>(mouse_prev_pos.x),
                              static_cast<float>(mouse_prev_pos.y) };
        const float2 to   = { static_cast<float>(x),
                              static_cast<float>(y) };

        const float2 a = { from.x / width, from.y / height };
        const float2 b = { to.x   / width, to.y   / height };

        camera_rotate = arcball.rotate( b, a );
    }

    mouse_prev_pos = make_int2( x, y );
}


void glutResize( int w, int h )
{
    if ( w == (int)width && h == (int)height ) return;

    width  = w;
    height = h;
    sutil::ensureMinimumSize(width, height);

    sutil::resizeBuffer( getOutputBuffer(), width, height );
    
    glViewport(0, 0, width, height);                                               

    glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
        "App Options:\n"
        "  -h | --help               Print this usage message and exit.\n"
        "  -f | --file               Save single frame to file and exit.\n"
        "  -n | --nopbo              Disable GL interop for display buffer.\n"
        "  -m | --mesh <mesh_file>   Specify path to mesh to be loaded.\n"
        "  -r | --report <LEVEL>     Enable usage reporting and report level [1-3].\n"
        "  -i | --ignore-materials   Ignore materials in the mesh file.\n"
        "       --no-triangle-api    Disable the Triangle API.\n"
        "App Keystrokes:\n"
        "  q  Quit\n" 
        "  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
        << std::endl;

    exit(1);
}

int main( int argc, char** argv )
 {
    std::string out_file;
    std::string mesh_file = std::string( sutil::samplesDir() ) + "/data/cow.obj";
    int usage_report_level = 0;
    for( int i=1; i<argc; ++i )
    {
        const std::string arg( argv[i] );

        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "-f" || arg == "--file"  )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            out_file = argv[++i];
        }
        else if( arg == "-n" || arg == "--nopbo"  )
        {
            use_pbo = false;
        }
        else if( arg == "-m" || arg == "--mesh" )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            mesh_file = argv[++i];
        }
        else if( arg == "-r" || arg == "--report" )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            usage_report_level = atoi( argv[++i] );
        }
        else if( arg == "-i" || arg == "--ignore-materials" )
        {
            ignore_mats = true;
        }
        else if( arg == "--no-triangle-api" )
        {
            use_tri_api = false;
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        glutInitialize( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif

        UsageReportLogger logger;
        createContext( usage_report_level, &logger );
        loadMesh( mesh_file );
        setupCamera();
        setupLights();

        context->validate();

        if ( out_file.empty() )
        {
            glutRun();
        }
        else
        {
            updateCamera();
            context->launch( 0, width, height );
            sutil::displayBufferPPM( out_file.c_str(), getOutputBuffer() );
            destroyContext();
        }
        return 0;
    }
    SUTIL_CATCH( context->get() )
}

