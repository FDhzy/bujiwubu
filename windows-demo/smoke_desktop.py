import sys
from launch_desktop import configure
configure()
sys.argv += ['--desktop-demo', '--debug']
from main import BlindNavigationApp, setup_logger, set_debug_mode
from kivy.clock import Clock
import numpy as np
from models.pytorch_shoe_model import PyTorchShoeModel
import logging
setup_logger(level=logging.INFO)
set_debug_mode(True)
app=BlindNavigationApp()
status={}
def verify(_):
 try:
  o=app.inference_engine.obstacle_detector.pytorch_model
  w=app.inference_engine.desktop_water
  assert o is not None and o.interpreter is not None
  assert w is not None and w.threshold == .6
  assert o.conf_threshold == .35
  app.main_screen.on_register_shoes()
  assert app.root.current == 'shoe_type_choice'
  app.shoe_type_choice_screen._return_to_main()
  assert app.root.current == 'main'
  original_start=app.camera_manager.start
  app.camera_manager.start=lambda: False
  try:
   app.main_screen.on_enable_camera()
   assert not app._processing_active and not app.main_screen._is_navigating
  finally: app.camera_manager.start=original_start
  shoe=PyTorchShoeModel(use_database=False)
  assert shoe.load_model('training_checkpoints/shoe_model.tflite')
  frame=np.random.default_rng(42).integers(0,256,(240,320,3),dtype=np.uint8)
  ob=o.detect(frame)
  wa=w.detect(frame)
  assert wa is not None and wa.mask.shape==frame.shape[:2]
  st,conf=shoe.predict(frame)
  assert 0 < conf <= 1
  res=app.inference_engine.process_frame(frame)
  assert res is not None
  app.main_screen.update_preview(frame)
  app.main_screen.update_debug_overlay(res,None)
  app.camera_manager.get_frame=lambda: frame
  app._processing_active=True
  app._process_frame(0)
  assert app.main_screen.ids.status_label.text.startswith('\u5ba4\u5185')
  # Inject a detector failure and verify it is visible in the UI.
  fn=app.inference_engine.obstacle_detector.detect
  app.inference_engine.obstacle_detector.detect=lambda _: None
  try:
   assert app.inference_engine.process_frame(frame) is None
   app._process_frame(0)
   assert app.main_screen.ids.status_label.text == '\u6a21\u578b\u68c0\u6d4b\u4e0d\u53ef\u7528'
  finally:
   app.inference_engine.obstacle_detector.detect=fn
   app._processing_active=False
  print('WINDOW_AND_INFERENCE_OK',app.root.current,len(ob),wa.mask.shape,st,conf,flush=True)
  status['ok']=True
 finally:
  app.stop()
Clock.schedule_once(verify,1)
app.run()
if not status.get('ok'): raise SystemExit(1)

