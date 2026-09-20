import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from ssvep_training.speller_ui import run_flicker
from ssvep_training.selection import trca_choice


class LiveTRCATests(unittest.TestCase):
    def test_policy_does_not_turn_invalid_data_into_b(self):
        self.assertEqual(trca_choice(0,.09,.1,'b'),1)
        self.assertEqual(trca_choice(0,.10,.1,'b'),0)
        self.assertEqual(trca_choice(1,.2,.1,'b'),1)
        self.assertEqual(trca_choice(0,.09,.1,'reject'),-1)
        self.assertEqual(trca_choice(0,.09,.1,'winner'),0)
        self.assertEqual(trca_choice(1,-.2,.1,'winner'),1)
        for choice, peak in [(-1,.4),(0,float('nan')),(1,float('inf')),(0,None)]:
            self.assertEqual(trca_choice(choice,peak,.1,'b'),-1)
            self.assertEqual(trca_choice(choice,peak,.1,'winner'),-1)

    def test_live_trca_ignores_cca_gate_but_obeys_quality_and_gestures(self):
        for peak, choice, late, gesture, expected in [
                (.2,0,False,False,[0]), (.05,0,False,False,[0]),
                (.05,1,False,False,[1]),
                (.2,1,False,False,[1]), (.2,-1,False,False,[]),
                (.05,0,True,False,[]), (.05,0,False,True,[])]:
            with self.subTest(peak=peak,choice=choice,late=late,gesture=gesture):
                ticks = iter(i*.1 for i in range(100))
                ui = NS(action_count=0, draws=0, picks=[], pick_suggestion=lambda *a: None,
                        set_progress=lambda *a: None)
                ui.draw = lambda: setattr(ui,'draws',ui.draws+1)
                ui.select_box = ui.picks.append
                recorder = NS(prediction_count=NS(value=0),last_sigma=NS(value=float('-inf')),
                              last_prediction=NS(value=choice),last_trca_peak=NS(value=peak),
                              gaze=NS(value=0), gesture_count=0,
                              cancel_prediction=lambda:None,prediction_is_current=lambda:True,
                              mark_onset=lambda *args:None,is_alive=lambda:True)
                recorder.read_gesture = lambda: (recorder.gesture_count,2)
                win = NS(nDroppedFrames=0,frameIntervals=[],ssvep_refresh=60,
                         flip=lambda:None,callOnFlip=lambda fn,*args:fn(*args))
                def request():
                    recorder.prediction_count.value += 1
                    win.nDroppedFrames += int(late)
                    recorder.gesture_count += int(gesture)
                recorder.request_prediction = request
                psychopy = NS(core=NS(Clock=lambda:NS(getTime=lambda:next(ticks))))
                stimulus = NS(draw_blank=lambda *a:None,flicker_frame=lambda *a:None)
                with patch.dict(sys.modules,{'psychopy':psychopy,'ssvep_training.stimulus':stimulus}), \
                     patch('ssvep_training.speller_ui.handle_keys',side_effect=lambda *a,**k:ui.draws<35):
                    run_flicker(win,ui,[],recorder,decoder='trca',trca_min_peak=.1,trca_below='winner')
                self.assertEqual(ui.picks,expected)


if __name__ == '__main__':
    unittest.main()
