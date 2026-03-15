import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the backend directory to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import importlib.util

# Import rve-backend.py as a module
spec = importlib.util.spec_from_file_location("rve_backend", os.path.abspath(os.path.join(os.path.dirname(__file__), 'rve-backend.py')))
rve_backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rve_backend)
HandleApplication = rve_backend.HandleApplication

class TestRVEBackend(unittest.TestCase):

    def setUp(self):
        # Mock sys.argv to simulate command-line arguments
        self.original_argv = sys.argv

    def tearDown(self):
        # Restore original sys.argv
        sys.argv = self.original_argv

    @patch('sys.exit')
    @patch('builtins.print')
    def test_version_option(self, mock_print, mock_exit):
        """Test the --version option."""
        sys.argv = ['rve-backend.py', '--version']
        with patch('rve_backend.HandleApplication.handleArguments', return_value=MagicMock(version=True, list_backends=False)):
            HandleApplication()
            mock_exit.assert_called_with(0)
            self.assertTrue(mock_print.called)

    @patch('rve_backend.BackendDetect')
    @patch('builtins.print')
    def test_list_backends_option(self, mock_print, mock_backend_detect):
        """Test the --list_backends option."""
        sys.argv = ['rve-backend.py', '--list_backends']
        
        # Mock the backend detection logic
        mock_backend_instance = mock_backend_detect.return_value
        mock_backend_instance.get_tensorrt.return_value = '8.2.5.1'
        mock_backend_instance.pytorch_device = 'cuda'
        mock_backend_instance.pytorch_version = '1.12.1'
        mock_backend_instance.get_ncnn.return_value = '20220729'
        mock_backend_instance.get_half_precision.return_value = True
        mock_backend_instance.get_gpus_torch.return_value = ['NVIDIA GeForce RTX 3080']
        mock_backend_instance.get_gpus_ncnn.return_value = ['NVIDIA GeForce RTX 3080']

        with patch('rve_backend.HandleApplication.handleArguments', return_value=MagicMock(list_backends=True, version=False)):
            app = HandleApplication()
            app.listBackends()
            self.assertTrue(mock_print.called)
            
    @patch('rve_backend.OpenCVInfo')
    @patch('rve_backend.print_video_info')
    @patch('sys.exit')
    def test_print_video_info_option(self, mock_exit, mock_print_video_info, mock_opencv_info):
        """Test the --print_video_info option."""
        sys.argv = ['rve-backend.py', '--print_video_info', 'dummy.mp4']
        with patch('rve_backend.HandleApplication.handleArguments', return_value=MagicMock(print_video_info='dummy.mp4', list_backends=False, version=False)):
            HandleApplication()
            mock_opencv_info.assert_called_with('dummy.mp4')
            mock_print_video_info.assert_called()
            mock_exit.assert_called_with(0)

    @patch('rve_backend.Render')
    @patch('rve_backend.download_ffmpeg')
    @patch('rve_backend.OpenCVInfo')
    @patch('os.path.isfile', return_value=True)
    def test_render_video_default_options(self, mock_isfile, mock_opencv_info, mock_download_ffmpeg, mock_render):
        """Test renderVideo with default options."""
        sys.argv = ['rve-backend.py', '-i', 'input.mp4', '-o', 'output.mp4']
        
        # Mocking the argument parser result
        args = MagicMock(
            input='input.mp4', output='output.mp4', interpolate_model=None, interpolate_factor=1.0,
            upscale_model=None, extra_restoration_models=None, tilesize=0, device='auto',
            backend='pytorch', precision='auto', pytorch_gpu_id=0, ncnn_gpu_id=0, start_time=None,
            end_time=None, overwrite=False, crf='18', video_encoder_preset='libx264',
            audio_encoder_preset='copy_audio', subtitle_encoder_preset='copy_subtitle',
            audio_bitrate='192k', benchmark=False, custom_encoder=None, border_detect=False,
            hdr_mode=False, video_pixel_format='yuv420p', merge_subtitles=True,
            pause_shared_memory_id=None, scene_detect_method='pyscenedetect',
            scene_detect_threshold=4.0, preview_shared_memory_id=None,
            tensorrt_opt_profile=3, tensorrt_dynamic_shapes=False, override_upscale_scale=None,
            UHD_mode=False, slomo_mode=False, dynamic_scaled_optical_flow=False,
            ensemble=False, output_to_mpv=False, list_backends=False, version=False,
            print_video_info=None
        )

        with patch('rve_backend.HandleApplication.handleArguments', return_value=args):
            app = HandleApplication()
            mock_download_ffmpeg.assert_called_once()
            mock_render.assert_called_once()

    @patch('rve_backend.Render')
    @patch('rve_backend.download_ffmpeg')
    @patch('rve_backend.OpenCVInfo')
    @patch('os.path.isfile', return_value=True)
    def test_all_render_options(self, mock_isfile, mock_opencv_info, mock_download_ffmpeg, mock_render):
        """Test that all options are passed correctly to the Render class."""
        sys.argv = [
            'rve-backend.py',
            '-i', 'input.mp4',
            '-o', 'output.mp4',
            '--start_time', '10.5',
            '--end_time', '20.5',
            '--backend', 'tensorrt',
            '--upscale_model', 'upscale.pth',
            '--extra_restoration_models', 'restore1.pth',
            '--extra_restoration_models', 'restore2.pth',
            '--interpolate_model', 'interp.pth',
            '--interpolate_factor', '2.0',
            '--precision', 'float16',
            '--tensorrt_opt_profile', '5',
            '--tensorrt_dynamic_shapes',
            '--scene_detect_method', 'mean',
            '--scene_detect_threshold', '2.0',
            '--overwrite',
            '--border_detect',
            '--crf', '22',
            '--video_encoder_preset', 'libx265',
            '--video_pixel_format', 'yuv422p10le',
            '--audio_encoder_preset', 'aac',
            '--subtitle_encoder_preset', 'srt',
            '--audio_bitrate', '256k',
            '--custom_encoder', 'my_encoder',
            '--tilesize', '512',
            '--device', 'cuda',
            '--pytorch_gpu_id', '1',
            '--ncnn_gpu_id', '1',
            '--benchmark',
            '--UHD_mode',
            '--slomo_mode',
            '--hdr_mode',
            '--dynamic_scaled_optical_flow',
            '--ensemble',
            '--preview_shared_memory_id', 'mem_id_preview',
            '--output_to_mpv',
            '--pause_shared_memory_id', 'mem_id_pause',
            '--merge_subtitles',
            '--override_upscale_scale', '1280',
        ]
        
        app = HandleApplication()
        
        mock_render.assert_called_once()
        called_args, _ = mock_render.call_args
        
        self.assertEqual(called_args['inputFile'], 'input.mp4')
        self.assertEqual(called_args['outputFile'], 'output.mp4')
        self.assertEqual(called_args['interpolateModel'], 'interp.pth')
        self.assertEqual(called_args['interpolateFactor'], 2.0)
        self.assertEqual(called_args['upscaleModel'], 'upscale.pth')
        self.assertEqual(called_args['extraRestorationModels'], ['restore1.pth', 'restore2.pth'])
        self.assertEqual(called_args['tile_size'], 512)
        self.assertEqual(called_args['device'], 'cuda')
        self.assertEqual(called_args['backend'], 'tensorrt')
        self.assertEqual(called_args['precision'], 'float16')
        self.assertEqual(called_args['pytorch_gpu_id'], 1)
        self.assertEqual(called_args['ncnn_gpu_id'], 1)
        self.assertEqual(called_args['start_time'], 10.5)
        self.assertEqual(called_args['end_time'], 20.5)
        self.assertTrue(called_args['overwrite'])
        self.assertEqual(called_args['crf'], '22')
        self.assertEqual(called_args['video_encoder_preset'], 'libx265')
        self.assertEqual(called_args['audio_encoder_preset'], 'aac')
        self.assertEqual(called_args['subtitle_encoder_preset'], 'srt')
        self.assertEqual(called_args['audio_bitrate'], '256k')
        self.assertTrue(called_args['benchmark'])
        self.assertEqual(called_args['custom_encoder'], 'my_encoder')
        self.assertTrue(called_args['border_detect'])
        self.assertTrue(called_args['hdr_mode'])
        self.assertEqual(called_args['pixelFormat'], 'yuv422p10le')
        self.assertTrue(called_args['merge_subtitles'])
        self.assertEqual(called_args['pause_shared_memory_id'], 'mem_id_pause')
        self.assertEqual(called_args['sceneDetectMethod'], 'mean')
        self.assertEqual(called_args['sceneDetectSensitivity'], 2.0)
        self.assertEqual(called_args['sharedMemoryID'], 'mem_id_preview')
        self.assertEqual(called_args['trt_optimization_level'], 5)
        self.assertTrue(called_args['trt_dynamic_shapes'])
        self.assertEqual(called_args['override_upscale_scale'], 1280)
        self.assertTrue(called_args['UHD_mode'])
        self.assertTrue(called_args['slomo_mode'])
        self.assertTrue(called_args['dynamic_scaled_optical_flow'])
        self.assertTrue(called_args['ensemble'])
        self.assertTrue(called_args['output_to_mpv'])

    @patch('rve_backend.HandleApplication.renderVideo')
    def test_batch_processing(self, mock_render_video):
        """Test batch processing from a text file."""
        app = HandleApplication.__new__(HandleApplication)
        
        # Mock initial arguments to point to a batch file
        app.args = MagicMock(input='batch.txt', list_backends=False, version=False, print_video_info=None)
        
        file_content = "-i video1.mp4 -o out1.mp4\n-i video2.mp4 -o out2.mp4 --crf 20"
        
        with patch('builtins.open', unittest.mock.mock_open(read_data=file_content)):
            with patch('rve_backend.HandleApplication.handleArguments') as mock_handle_args:
                # Simulate reparsing arguments for each line in the batch file
                def arg_side_effect(*args, **kwargs):
                    if 'video1' in sys.argv:
                        return MagicMock(input='video1.mp4', output='out1.mp4', crf='18', list_backends=False, version=False, print_video_info=None)
                    elif 'video2' in sys.argv:
                        return MagicMock(input='video2.mp4', output='out2.mp4', crf='20', list_backends=False, version=False, print_video_info=None)
                    return MagicMock()
                
                mock_handle_args.side_effect = arg_side_effect
                
                result = app.batchProcessing()

                self.assertTrue(result)
                self.assertEqual(mock_render_video.call_count, 2)

    def test_check_arguments_exceptions(self):
        """Test argument validation logic."""
        app = HandleApplication.__new__(HandleApplication)
        
        # Test output exists error
        with self.assertRaises(os.error):
            app.args = MagicMock(output='existing_file.mp4', overwrite=False, benchmark=False, input='input.mp4')
            with patch('os.path.isfile', side_effect=[True, True]): # mock output and input file checks
                app.checkArguments()

        # Test input not exists error
        with self.assertRaises(os.error):
            app.args = MagicMock(output='output.mp4', overwrite=True, input='non_existing_file.mp4', benchmark=False)
            with patch('os.path.isfile', return_value=False):
                app.checkArguments()

        # Test invalid tilesize
        with self.assertRaises(ValueError):
            app.args = MagicMock(tilesize=-1, input='input.mp4', output='output.mp4', overwrite=True, benchmark=False)
            with patch('os.path.isfile', return_value=True):
                app.checkArguments()
                
        # Test invalid interpolation factor
        with self.assertRaises(ValueError):
            app.args = MagicMock(interpolate_factor=-1, input='input.mp4', output='output.mp4', overwrite=True, benchmark=False, tilesize=0)
            with patch('os.path.isfile', return_value=True):
                app.checkArguments()

        # Test interpolation factor > 1 with no model
        with self.assertRaises(ValueError):
            app.args = MagicMock(interpolate_factor=2, interpolate_model=None, input='input.mp4', output='output.mp4', overwrite=True, benchmark=False, tilesize=0)
            with patch('os.path.isfile', return_value=True):
                app.checkArguments()

        # Test interpolation factor == 1 with model
        with self.assertRaises(ValueError):
            app.args = MagicMock(interpolate_factor=1, interpolate_model='rife46', input='input.mp4', output='output.mp4', overwrite=True, benchmark=False, tilesize=0)
            with patch('os.path.isfile', return_value=True):
                app.checkArguments()
