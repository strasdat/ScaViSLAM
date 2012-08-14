// This file is part of ScaViSLAM.
//
// Copyright 2011 Hauke Strasdat (Imperial College London)
//
// ScaViSLAM is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// any later version.
//
// ScaViSLAM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with ScaViSLAM.  If not, see <http://www.gnu.org/licenses/>.

#ifdef SCAVISLAM_CUDA_SUPPORT
#include <cutil_inline.h>
#endif

#include <pangolin/pangolin.h>

#include <visiontools/accessor_macros.h>
#include <visiontools/draw2d.h>
#include <visiontools/draw3d.h>
#include <visiontools/performance_monitor.h>

#include "global.h"

#include "backend.h"
#include "stereo_camera.h"
#include "draw_items.h"
#include "frame_grabber.hpp"
#include "placerecognizer.h"
#include "stereo_frontend.h"


using namespace ScaViSLAM;

bool show_next_frame = false;
bool play_frames = false;
bool stop = false;


class PyramidView
{
public:
  static vector<pangolin::View *>
  init(size_t num_levels,
       double top,
       double bottom,
       double left,
       double right,
       const std::string & base,
       double ratio=4./3.)
  {
    vector<pangolin::View *> pyr;
    float w = right-left;
    float h = (top-bottom)*0.5;
    float y_offset = h;

    for (size_t l=0; l<num_levels; ++l)
    {
      std::stringstream ss;
      ss << base << " " << l;
      pangolin::View & v = pangolin::Display(ss.str())
          .SetBounds(bottom+h+y_offset,
                     bottom+y_offset,
                     left,
                     left+w,
                     ratio);
      pyr.push_back(&v);
      w*=0.5;
      h*=0.5;
      y_offset = h;
    }
    return pyr;
  }
};

struct Views
{
  pangolin::OpenGlRenderState pangolin_cam;
  vector<pangolin::View*> pyr;
  vector<pangolin::View*> key_pyr;
  pangolin::View * right;
  pangolin::View * v3d;
  pangolin::View * panel;
  pangolin::Plotter * graph;
  pangolin::DataLog logger;
};

struct Modules
{
  Modules()
    : frame_grabber(NULL),
      frontend(NULL),
      backend(NULL),
      per_mon(NULL),
      pla_reg(NULL) {}

  ~Modules()
  {
    if(frame_grabber!=NULL)
      delete frame_grabber;
    if(frontend!=NULL)
      delete frontend;
    if(backend!=NULL)
      delete backend;
    if(per_mon!=NULL)
      delete per_mon;
    if(pla_reg!=NULL)
      delete pla_reg;
  }

  FrameGrabber<StereoCamera> * frame_grabber;
  StereoFrontend * frontend;
  Backend * backend;
  PerformanceMonitor * per_mon;
  PlaceRecognizer * pla_reg;
};

Views
initializeViews(const StereoCamera & stereo_camera)
{
  pangolin::CreateGlutWindowAndBind("Main",
                                    stereo_camera.width()*2.5,
                                    stereo_camera.height()*2);
  Views views;
  views.pyr
      = PyramidView::init(NUM_PYR_LEVELS,
                          1.0,
                          0.0,
                          0.2,
                          0.6,
                          "left");
  views.right
      = &(pangolin::Display("right")
          .SetBounds(1.0,0.75,0.6,0.8,4./3.));
  views.key_pyr
      = PyramidView::init(NUM_PYR_LEVELS,
                          1.0,0.5,0.8,1.,
                          "key");
  views.pangolin_cam.Set(stereo_camera.getOpenGlMatrixSpec());
  views.pangolin_cam.Set(pangolin::IdentityMatrix(pangolin::GlModelViewStack));

  views.v3d = &(pangolin::Display("view3d")
                .SetBounds(0.5,0.0,0.6,1.0,true)
                .SetHandler(new pangolin::Handler3D(views.pangolin_cam)));

  views.panel = &(pangolin::CreatePanel("ui")
                  .SetBounds(1.0, 0.0, 0, 0.2));

  views.graph = &(pangolin::CreatePlotter("x",&(views.logger)));
  views.graph->int_x[1] = views.graph->int_x_dflt[1] = 50;
  views.graph->int_y[0] = views.graph->int_y_dflt[0] = 0;
  views.graph->ticks[1] = 0.05;
  views.graph->plot_mode = pangolin::Plotter::STACKED_HISTOGRAM;
  views.graph->SetBounds(0.5,0.0,0.4,0.6,false);
  return views;
}

Modules
startModules(const StereoCamera & stereo_camera,
             Views * views)
{
  Modules modules;
  modules.per_mon = new PerformanceMonitor;
  modules.frame_grabber
      = new FrameGrabber<StereoCamera>(stereo_camera,
                                       Vector4d(0,0,0,0),
                                       modules.per_mon);
  modules.per_mon->add("drawing");
  modules.per_mon->add("back end");
  modules.per_mon->add("grab frame");
  modules.per_mon->add("preprocess");
  modules.per_mon->add("stereo");
  modules.per_mon->add("dense tracking");
  modules.per_mon->add("fast");
  modules.per_mon->add("match");
  modules.per_mon->add("process points");
  modules.per_mon->add("drop keyframe");
  modules.per_mon->add("dense point cloud");
  modules.per_mon->setup(&views->logger);

  modules.frame_grabber->initialise();
  modules.frontend =
      new StereoFrontend(&modules.frame_grabber->frame_data, modules.per_mon);
  modules.frontend->initialize();
  modules.pla_reg = new PlaceRecognizer(stereo_camera);
  modules.backend
      = new Backend(modules.frame_grabber->frame_data.cam_vec,
                    &modules.pla_reg->monitor);
  boost::thread place_reg_thread(boost::ref(*modules.pla_reg));
  boost::thread backend_thread(boost::ref(*modules.backend));

  static pangolin::Var<bool> livestream
      ("framepipe.livestream",false);

  if (livestream)
  {
    for (int i=0; i<1000; ++i)
    {
      modules.frame_grabber->processNextFrame();
      cerr << i << endl;
    }
  }
  modules.frame_grabber->processNextFrame();
  modules.frontend->processFirstFrame();
  assert(modules.frontend->to_optimizer_stack.size()==1);
  modules.backend->monitor
      .pushKeyframe(modules.frontend->to_optimizer_stack.top());
  modules.frontend->to_optimizer_stack.pop();
  return modules;
}

//TODO: method way too long...
void draw(int loop_id,
          Views * views,
          Modules * modules)
{
  modules->per_mon->start("drawing");
  GLUquadric * quad = gluNewQuadric();
  SE3 T_actkey_from_world;
  if (IS_IN_SET(modules->frontend->actkey_id,
                modules->frontend->neighborhood()->vertex_map))
  {
    T_actkey_from_world
        = GET_MAP_ELEM(modules->frontend->actkey_id,
                       modules->frontend->neighborhood()->vertex_map)
        .T_me_from_w;
  }
  static pangolin::Var<bool>
      ui_show_new_points("ui.show_new_points",true);
  static pangolin::Var<bool>
      ui_show_tracked_points("ui.show_tracked_points",true);
  static pangolin::Var<bool>
      ui_show_newtracked_points("ui.show_newtracked_points",true);
  static pangolin::Var<bool>
      ui_show_optimized_points("ui.show_optimized_points",true);
  static pangolin::Var<bool>
      ui_show_fast_points("ui.show_fast_points",false);

  static pangolin::Var<bool>
      ui_show_neighborhood("ui.show_neighborhood", false);
  static  pangolin::Var<int> ui_debug
      ("ui.debug",1,0,6);
  static  pangolin::Var<int> ui_debug_level
      ("ui.debug_level",0,0,NUM_PYR_LEVELS-1);

  static pangolin::Var<int>
      ui_show_keyframe("ui.show_keyframe",0,0,
                       modules->frontend->keyframe_num2id.size()-1);
  ui_show_keyframe.var->meta_range[0] = 0;
  ui_show_keyframe.var->meta_range[1]
      = modules->frontend->keyframe_num2id.size()-1;

  if (loop_id>-1)
  {
    ui_show_keyframe
        = GET_MAP_ELEM(loop_id, modules->frontend->keyframe_id2num);
  }



  cv::Mat debug;
#ifdef SCAVISLAM_CUDA_SUPPORT
  if (ui_debug==0)
  {
    modules->frontend->tracker()
        .dev_residual_img[ui_debug_level].download(debug);
  }
  else if (ui_debug==1)
  {
    modules->frame_grabber->frame_data.cur_left()
        .gpu_pyr_float32[ui_debug_level].download(debug);
  }
  else if (ui_debug==2)
  {
    modules->frame_grabber->frame_data.prev_left()
        .gpu_pyr_float32[ui_debug_level].download(debug);
  }
  else if (ui_debug==3)
  {
    modules->frame_grabber->frame_data
        .gpu_pyr_float32_dx[ui_debug_level].download(debug);
  }
  else if (ui_debug==4)
  {
    modules->frame_grabber->frame_data
        .gpu_pyr_float32_dy[ui_debug_level].download(debug);
  }
  else if (ui_debug==5)
  {
    debug = modules->frame_grabber->frame_data.right.uint8;
  }
  else
  {
    debug = modules->frame_grabber->frame_data.color_disp;
  }
#else
  if (ui_debug==0)
  {
    debug = modules->frontend->tracker().residual_img[ui_debug_level];
  }
  else if (ui_debug==1)
  {
    debug = modules->frame_grabber->frame_data.pyr_float32[ui_debug_level];
  }
  else if (ui_debug==2)
  {
    debug
        = modules->frame_grabber
        ->frame_data.prev_left().pyr_uint8[ui_debug_level];
  }
  else if (ui_debug==3)
  {
    debug = modules->frame_grabber->frame_data.pyr_float32_dx[ui_debug_level];
  }
  else if (ui_debug==4)
  {
    debug = modules->frame_grabber->frame_data.pyr_float32_dy[ui_debug_level];
  }
  else if (ui_debug==5)
  {
    debug = modules->frame_grabber->frame_data.right.uint8;
  }
  else
  {
    debug = modules->frame_grabber->frame_data.color_disp;
  }
#endif

  static BackendDrawDataPtr graph_draw_data(new BackendDrawData);
  modules->backend->monitor.getDrawData(&graph_draw_data);



  for (int level = 0; level<NUM_PYR_LEVELS; ++level)
  {
    views->pyr[level]->ActivateScissorAndClear();
    Draw2d::activate(modules->frame_grabber
                     ->frame_data.cam_vec[level].image_size());
    glColor3f(1,1,1);
    Draw2d::texture(modules->frame_grabber
                    ->frame_data.cur_left().pyr_uint8.at(level));
    if (ui_show_tracked_points)
    {
      glColor3f(0,0,1);
      const DrawItems::Line2dList & line_list
          = modules->frontend->draw_data().tracked_points2d.at(level);
      for (DrawItems::Line2dList::const_iterator it=line_list.begin();
           it!=line_list.end(); ++it)
      {
        DrawItems::Line2d line = *it;
        Draw2d::line(line.p1,  line.p2);
        Draw2d::circle(line.p2,0,2,quad);
      }
    }
    if (ui_show_newtracked_points)
    {
      glColor3f(0.5,0.5,1);
      const DrawItems::Line2dList & line_list
          = modules->frontend->draw_data().newtracked_points2d.at(level);
      for (DrawItems::Line2dList::const_iterator it=line_list.begin();
           it!=line_list.end(); ++it)
      {
        DrawItems::Line2d line = *it;
        Draw2d::line(line.p1,
                     line.p2);

        Draw2d::circle(line.p2,0,2,quad);
      }
    }
    if (ui_show_new_points)
    {
      glColor3f(0,1,0);
      Draw2d::points(modules->frontend->draw_data().new_points2d.at(level),2);
      glColor3f(0,0,1);
      Draw2d::points(modules->frontend->draw_data().fast_points2d.at(level),5);
    }

    if (level==0)
    {
      glColor3f(1,0,1);
      for (DrawItems::CircleList::const_iterator it =
           modules->frontend->draw_data().blobs2d.begin();
           it!= modules->frontend->draw_data().blobs2d.end(); ++it)
      {
        const DrawItems::Circle & c = *it;
        Draw2d::circle(c.center,c.radius-1,c.radius+1,quad);
      }
    }
    if (ui_show_fast_points)
    {
      glColor3f(0,1,1);
      Draw2d::points(modules->frontend->draw_data().fast_points2d.at(level),2);
    }
  }

  views->right->ActivateScissorAndClear();
  Draw2d::activate(debug.size());
  glColor3f(1,1,1);
  Draw2d::texture(debug);

  for (int l = 0; l<NUM_PYR_LEVELS; ++l)
  {
    views->key_pyr[l]->ActivateScissorAndClear();
    Draw2d::activate(modules->frame_grabber
                     ->frame_data.cam_vec.at(l).image_size());
    int show_id = ui_show_keyframe;

    int keyframe_id = modules->frontend->keyframe_num2id.at(show_id);
    tr1::unordered_map<int,Frame>::const_iterator it
        = modules->frontend->keyframe_map.find(keyframe_id);
    if(it != modules->frontend->keyframe_map.end())
    {
      if (it->second.pyr.size()>(size_t)l)
      {
        if (loop_id>-1)
          glColor3f(0.3,0.3,1);
        else
          glColor3f(1,1,1);
        Draw2d::texture(it->second.pyr[l]);
        if (ui_show_tracked_points)
        {
          const ALIGNED<DrawItems::Point2dVec>::int_hash_map & keymap
              = modules->frontend->draw_data().tracked_anchorpoints2d.at(l);
          ALIGNED<DrawItems::Point2dVec>::int_hash_map::const_iterator it2
              = keymap.find(keyframe_id);
          if (it2!=keymap.end())
          {
            const DrawItems::Point2dVec & pointvec = it2->second;
            glColor3f(0,0,1);
            Draw2d::points(pointvec,  2);
          }
        }
      }
    }
  }
  SE3 T_cur_from_world
      = modules->frontend->T_cur_from_actkey()*T_actkey_from_world;
  const pangolin::OpenGlMatrix & gl_projection =
       views->pangolin_cam.GetProjectionMatrix();
  pangolin::OpenGlMatrix gl_modelview =
                        views->pangolin_cam.GetModelViewMatrix();
  Map<Matrix<double,4,4,ColMajor> > Map_gl_modelview(&gl_modelview.m[0]);

  Map_gl_modelview
      = Map_gl_modelview*T_cur_from_world.matrix();

  pangolin::OpenGlRenderState r_state;
  r_state.SetProjectionMatrix(gl_projection);
  r_state.SetModelViewMatrix(gl_modelview);
  views->v3d->ActivateScissorAndClear(r_state);
  glClearColor(1,1,1,0);
  glEnable(GL_DEPTH_TEST);
  glColor3f(0,0,0);
  glColor3f(0.75,0,0);
  if (ui_show_neighborhood)
  {
    glColor4f(1,0,0,0.5);

    vector<GlPoint3f> points;
    for (list<CandidatePoint3Ptr>::const_iterator
         it=modules->frontend->neighborhood()->point_list.begin();
         it!=modules->frontend->neighborhood()->point_list.end(); ++it)
    {
      const CandidatePoint3Ptr & ap = *it;
      points.push_back(
            GlPoint3f(GET_MAP_ELEM(ap->anchor_id,
                                   modules->frontend->neighborhood()
                                   ->vertex_map).T_me_from_w.inverse()
                      *ap->xyz_anchor));
    }
    Draw3d::points(points,2);
    for (ALIGNED<FrontendVertex>::int_hash_map::const_iterator
         it=modules->frontend->neighborhood()->vertex_map.begin();
         it!=modules->frontend->neighborhood()->vertex_map.end(); ++it)
    {
      SE3 T1 = it->second.T_me_from_w.inverse();
      Draw3d::pose(T1);

      for (multimap<int,int>::const_iterator it2 =
           it->second.strength_to_neighbors.begin();
           it2 != it->second.strength_to_neighbors.end(); ++it2)
      {
        SE3 T2 = GET_MAP_ELEM(it2->second,
                              modules->frontend->neighborhood()->vertex_map)
            .T_me_from_w.inverse();
        Draw3d::line(T1.translation(), T2.translation());
      }
    }
  }
  glColor3f(0.75,0,0);
  if (ui_show_optimized_points)
  {
    for (StereoGraph::EdgeTable::const_iterator it
         =  graph_draw_data->new_edges.begin();
         it!= graph_draw_data->new_edges.end(); ++it)
    {
      glColor4f(0.75,0,0.,0.5);
      const StereoGraph::EdgePtr & e = it->second;
      int id1 = e->vertex_id1;
      int id2 = e->vertex_id2;


      if(IS_IN_SET(id1,graph_draw_data->double_window)==false
         || IS_IN_SET(id2,graph_draw_data->double_window)==false)
        continue;
      SE3 T_c1_from_w
          = GET_MAP_ELEM(id1,graph_draw_data->vertex_table).T_me_from_world;
      SE3 T_c2_from_w
          = GET_MAP_ELEM(id2,graph_draw_data->vertex_table).T_me_from_world;

      Draw3d::line(T_c1_from_w.inverse().translation(),
                   T_c2_from_w.inverse().translation());
    }
    for (StereoGraph::EdgeTable::const_iterator it
         =  graph_draw_data->edge_table.begin();
         it!= graph_draw_data->edge_table.end(); ++it)
    {
      const StereoGraph::EdgePtr & e = it->second;
      if (!e->is_marginalized())
        glColor4f(0.75,0,0,0.5);
      else  glColor4f(0.75,0.75,0.75,0.5);

      if (e->error>0.0000001)
      {
        glColor4f(0,0,1.,0.75);
      }

      int id1 = e->vertex_id1;
      int id2 = e->vertex_id2;

      if(IS_IN_SET(id1,graph_draw_data->double_window)==false
         || IS_IN_SET(id2,graph_draw_data->double_window)==false)
        continue;
      SE3 T_c1_from_w
          = GET_MAP_ELEM(id1,graph_draw_data->vertex_table).T_me_from_world;
      SE3 T_c2_from_w
          = GET_MAP_ELEM(id2,graph_draw_data->vertex_table).T_me_from_world;

      Draw3d::line(T_c1_from_w.inverse().translation(),
                   T_c2_from_w.inverse().translation());
    }
    for (StereoGraph::WindowTable::const_iterator
         it_win1 = graph_draw_data->double_window.begin();
         it_win1!=graph_draw_data->double_window.end();
         ++it_win1)
    {
      int pose_id_1 = it_win1->first;
      StereoGraph::WindowType wtype_1 = it_win1->second;
      if (wtype_1==StereoGraph::INNER)
        glColor3f(1,0,0);
      else
        glColor3f(0.5,0.5,0.5);

      const StereoGraph::Vertex & v
          = GET_MAP_ELEM(pose_id_1, graph_draw_data->vertex_table);

      Draw3d::pose(v.T_me_from_world.inverse());
    }
  }
  if (ui_show_tracked_points)
  {
    glColor4f(0,0,1,0.5);
    for (int l = 0; l<NUM_PYR_LEVELS; ++l)
    {
      Draw3d::points(modules->frontend->draw_data().tracked_points3d.at(l),
                     zeroFromPyr_d(2.,l));
    }

  }
  if (ui_show_newtracked_points)
  {
    glColor4f(0.5,0.5,1,0.5);
    for (int l = 0; l<NUM_PYR_LEVELS; ++l)
    {
      Draw3d::points(modules->frontend->draw_data().newtracked_points3d.at(l),
                     zeroFromPyr_d(2.,l));
    }
    Draw3d::pose((modules->frontend->T_cur_from_actkey() *
                  T_actkey_from_world
                  ).inverse());
  }
  if (ui_show_new_points)
  {
    glColor3f(0,1,0);
    for (int l = 0; l<NUM_PYR_LEVELS; ++l)
    {
      Draw3d::points(modules->frontend->draw_data().new_points3d.at(l),
                     T_actkey_from_world.inverse(),
                     zeroFromPyr_d(2.,l));
    }
    Draw3d::pose(
          T_actkey_from_world.inverse());
  }
  glColor3f(0,0,1);
  SE3 T_world_from_cur = (modules->frontend->T_cur_from_actkey() *
                          T_actkey_from_world
                          ).inverse();
  Draw3d::pose(T_world_from_cur,0.25);
  Draw3d::line(T_world_from_cur.translation(),
               T_actkey_from_world.inverse().translation());

  if (ui_show_optimized_points)
  {
    vector<vector<GlPoint3f> > glpoint_vec(NUM_PYR_LEVELS);


    for (tr1::unordered_set<int>::iterator
         it = graph_draw_data->active_point_set.begin();
         it!=graph_draw_data->active_point_set.end();++it)
    {
      int point_id = *it;
      const StereoGraph::Point & p
          = GET_MAP_ELEM(point_id, graph_draw_data->point_table);

      SE3 T_world_from_anchor
          = GET_MAP_ELEM(p.anchorframe_id, graph_draw_data->vertex_table)
          .T_me_from_world.inverse();

      glpoint_vec.at(p.anchor_level).push_back(
            GlPoint3f(T_world_from_anchor*p.xyz_anchor));

    }
    for (int l = 0; l<NUM_PYR_LEVELS; ++l)
    {
      Draw3d::points(glpoint_vec.at(l),
                     zeroFromPyr_d(2.,l));
    }
  }
  modules->per_mon->stop("drawing");
  modules->per_mon->plot(&views->logger);
  views->graph->Render();
  views->panel->Render();
  glutSwapBuffers();
  glutMainLoopEvent();
  gluDeleteQuadric(quad);
}


//TODO: main method to long...
int main(int argc, const char* argv[])
{
  if (argc<2)
  {
    cout << "please specify configuration file!" << endl;
    exit(0);
  }

  pangolin::ParseVarsFile(argv[1]);
  pangolin::Var<int> cam_width("cam.width",640);
  pangolin::Var<int> cam_height("cam.height",480);
  pangolin::Var<double> cam_f("cam.f",570.342);
  pangolin::Var<double> cam_px("cam.px",320);
  pangolin::Var<double> cam_py("cam.py",240);
  pangolin::Var<double> cam_baseline("cam.baseline",0.075);

#ifdef SCAVISLAM_CUDA_SUPPORT
  cudaDeviceProp prop;
  CUDA_SAFE_CALL( cudaGetDeviceProperties(&prop, 0) );
  std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
#endif

  StereoCamera stereo_camera((double)cam_f,
                             Vector2d((double)cam_px,(double)cam_py),
                             cv::Size((int)cam_width,(int)cam_height),
                             cam_baseline);

  Views views = initializeViews(stereo_camera);

  Modules modules = startModules(stereo_camera,
                                 &views);
  pangolin::Var<bool> next_button("ui.next Button",false,false);
  pangolin::Var<bool> play_button("ui.play Play",false,false);
  pangolin::Var<float> ui_fps("ui.fps",false);

  while(true)
  {
    if (!stop && (modules.frame_grabber->params().livestream
                  || show_next_frame) )
    {
      modules.per_mon->new_frame();
      ui_fps = modules.per_mon->fps();
      modules.frame_grabber->processNextFrame();
      NeighborhoodPtr neighborhood;
      modules.backend->monitor.queryNeighborhood(modules.frontend->actkey_id);
      static pangolin::Var<bool> updated("ui.updated",false,true);
      updated = false;

      if (modules.backend->monitor.getNeighborhood(&neighborhood))
      {
        if (neighborhood->vertex_map.find(modules.frontend->actkey_id)
            != neighborhood->vertex_map.end())
        {
          modules.frontend->neighborhood() = neighborhood;
          updated = true;

        }
      }
      bool is_frame_droped = false;
      bool tracking_worked = modules.frontend->processFrame(&is_frame_droped);
      if(tracking_worked==false)
      {
        cerr << "FAILURE!" << endl;
        exit(0);
      }
      if (is_frame_droped)
      {
        assert(modules.frontend->to_optimizer_stack.size()==1);
        AddToOptimzerPtr to_opt = modules.frontend->to_optimizer_stack.top();
        modules.backend->monitor.pushKeyframe(to_opt);
        modules.frontend->to_optimizer_stack.pop();
      }
    }
    int best_match = -1;
    DetectedLoop loop;
    bool is_loop_detected = modules.backend->monitor.getClosedLoop(&loop);
    if (is_loop_detected)
    {
      best_match = loop.loop_keyframe_id;
    }
    if (pangolin::ShouldQuit())
      exit(0);
    if(pangolin::HasResized())
      pangolin::DisplayBase().ActivateScissorAndClear();

    draw(best_match, &views, &modules);


    show_next_frame = false;
    if (pangolin::Pushed(next_button))
    {
      show_next_frame = true;
    }
    if (pangolin::Pushed(play_button))
    {

      play_frames = !play_frames;
    }
    if(play_frames)
      show_next_frame = true;
  }
  return 0;
}

