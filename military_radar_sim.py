import cv2
import numpy as np
import time
import math

class RadarTrack:
    def __init__(self, id, init_point):
        self.id = id
        self.active = True
        self.last_seen = time.time()
        
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.array([[1,0,0,0], [0,1,0,0], [0,0,5,0], [0,0,0,5]], np.float32) * 0.03
        
        self.kf.statePre = np.array([[init_point[0]], [init_point[1]], [0], [0]], np.float32)
        self.kf.correct(np.array([[init_point[0]], [init_point[1]]], np.float32))
        
        self.prediction = init_point
        self.history = []
        
        self.range = 0.0
        self.azimuth = 0.0
        self.velocity = 0.0
        self.threat_level = 0.0

    def update(self, measurement=None):
        prediction = self.kf.predict()
        self.prediction = (float(prediction[0][0]), float(prediction[1][0]))
        
        if measurement:
            meas_array = np.array([[measurement[0]], [measurement[1]]], np.float32)
            self.kf.correct(meas_array)
            self.last_seen = time.time()
        
        vx = float(self.kf.statePost[2][0])
        vy = float(self.kf.statePost[3][0])
        self.velocity = math.sqrt(vx*vx + vy*vy) * 10 
        
        self.history.append(self.prediction)
        if len(self.history) > 50: 
            self.history.pop(0)

class RadarSystem:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        
        self.width, self.height = 1280, 720
        self.radar_center = (400, 360)
        self.radar_radius = 300
        
        self.tracks = {}
        self.next_id = 101
        self.sweep_angle = 0
        
        self.no_fly_zone = np.array([
            [self.radar_center[0], self.radar_center[1]],
            [self.radar_center[0] - 150, self.radar_center[1] - 250],
            [self.radar_center[0] + 150, self.radar_center[1] - 250]
        ], np.int32)
        
        self.log_file = open("mission_log.csv", "w")
        self.log_file.write("TIMESTAMP,ID,AZIMUTH,RANGE,VELOCITY,THREAT_LEVEL\n")
        self.start_time = time.time()

    def calculate_polar(self, x, y, frame_w, frame_h):
        dx = x - frame_w/2
        dy = frame_h - y 
        azimuth = math.degrees(math.atan2(dx, dy))
        rng = math.sqrt(dx*dx + dy*dy)
        max_cam_range = math.sqrt((frame_w/2)**2 + frame_h**2)
        norm_range = (rng / max_cam_range) * self.radar_radius
        return azimuth, norm_range

    def draw_interface(self, screen, alarm_active=False):
        screen[:] = (10, 10, 20) 
        if alarm_active and int(time.time() * 5) % 2 == 0:
            screen[:] = (10, 10, 50) 

        cx, cy = self.radar_center
        rr = self.radar_radius
        
        cv2.polylines(screen, [self.no_fly_zone], True, (0, 0, 100), 2)
        cv2.fillPoly(screen, [self.no_fly_zone], (0, 0, 40)) 
        cv2.putText(screen, "RESTRICTED AIRSPACE", (cx - 60, cy - 100), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 200), 1)
        
        for r in range(1, 5):
            cv2.circle(screen, (cx, cy), int(rr * r/4), (0, 50, 0), 1)
        cv2.line(screen, (cx-rr, cy), (cx+rr, cy), (0, 50, 0), 1)
        cv2.line(screen, (cx, cy-rr), (cx, cy+rr), (0, 50, 0), 1)
        
        sweep_rad = math.radians(self.sweep_angle - 90)
        ex = int(cx + rr * math.cos(sweep_rad))
        ey = int(cy + rr * math.sin(sweep_rad))
        cv2.line(screen, (cx, cy), (ex, ey), (0, 200, 0), 2)
        self.sweep_angle = (self.sweep_angle + 3) % 360

        sidebar_x = 800
        cv2.rectangle(screen, (sidebar_x, 0), (self.width, self.height), (20, 20, 30), -1)
        
        status_color = (0, 255, 0)
        status_text = "NORMAL OPERATIONAL"
        if alarm_active:
            status_color = (0, 0, 255)
            status_text = "** AIRSPACE VIOLATION **"
            
        cv2.putText(screen, f"STATUS: {status_text}", (sidebar_x+20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        y_off = 100
        headers = ["ID", "AZIMUTH", "RANGE", "VEL", "THREAT"]
        x_positions = [0, 60, 160, 240, 320]
        for i, h in enumerate(headers):
            cv2.putText(screen, h, (sidebar_x + 20 + x_positions[i], y_off), cv2.FONT_HERSHEY_PLAIN, 0.8, (150, 150, 150), 1)
        cv2.line(screen, (sidebar_x+10, y_off+10), (self.width-10, y_off+10), (100, 100, 150), 1)
        
        return y_off + 30

    def run(self):
        cv2.namedWindow("ADX-C2 STATION", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ADX-C2 STATION", 1280, 720)
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            small_frame = cv2.resize(frame, (320, 240))
            small_frame = cv2.flip(small_frame, 1)
            mask = self.bg_subtractor.apply(small_frame)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            measurements = []
            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    x, y, w, h = cv2.boundingRect(cnt)
                    measurements.append((x + w/2, y + h/2))
            
            matched_indices = []
            for m in measurements:
                best_id = -1; min_dist = 50.0
                for tid, track in self.tracks.items():
                    dist = math.sqrt((track.prediction[0]-m[0])**2 + (track.prediction[1]-m[1])**2)
                    if dist < min_dist: min_dist = dist; best_id = tid
                if best_id != -1:
                    self.tracks[best_id].update(m)
                    matched_indices.append(best_id)
                else:
                    self.tracks[self.next_id] = RadarTrack(self.next_id, m)
                    self.next_id += 1
            
            for tid, track in self.tracks.items():
                if tid not in matched_indices: track.update(None)
            
            curr_time = time.time()
            self.tracks = {tid: t for tid, t in self.tracks.items() if (curr_time - t.last_seen) < 2.0}
            
            alarm_state = False
            
            for tid, track in self.tracks.items():
                azi_deg, rng_px = self.calculate_polar(track.prediction[0], track.prediction[1], 320, 240)
                rad_ang = math.radians(azi_deg - 90)
                rx = int(self.radar_center[0] + rng_px * math.cos(rad_ang))
                ry = int(self.radar_center[1] + rng_px * math.sin(rad_ang))
                
                dist_to_zone = cv2.pointPolygonTest(self.no_fly_zone, (rx, ry), False)
                if dist_to_zone >= 0: 
                    alarm_state = True

            display = np.zeros((self.height, self.width, 3), np.uint8)
            list_y = self.draw_interface(display, alarm_state)
            
            idx = 0
            for tid, track in self.tracks.items():
                azi_deg, rng_px = self.calculate_polar(track.prediction[0], track.prediction[1], 320, 240)
                rad_ang = math.radians(azi_deg - 90)
                rx = int(self.radar_center[0] + rng_px * math.cos(rad_ang))
                ry = int(self.radar_center[1] + rng_px * math.sin(rad_ang))
                
                color = (0, 255, 255)
                threat = "UNK"
                
                if cv2.pointPolygonTest(self.no_fly_zone, (rx, ry), False) >= 0:
                    color = (0, 0, 255) 
                    threat = "HOSTILE"
                
                vx = float(track.kf.statePost[2][0])
                vy = float(track.kf.statePost[3][0])
                
                cv2.circle(display, (rx, ry), 5, color, -1)
                cv2.line(display, (rx, ry), (int(rx+vx*2), int(ry+vy*2)), color, 1)
                cv2.putText(display, f"{str(tid)[-3:]}", (rx+5, ry-5), cv2.FONT_HERSHEY_PLAIN, 0.8, (200, 200, 200), 1)
                
                if idx < 12:
                    y = list_y + idx * 25
                    cv2.putText(display, f"{tid}", (820, y), cv2.FONT_HERSHEY_PLAIN, 1, (200,200,200), 1)
                    cv2.putText(display, f"{int(azi_deg)}", (880, y), cv2.FONT_HERSHEY_PLAIN, 1, (200,200,200), 1)
                    cv2.putText(display, f"{int(rng_px*10)}", (980, y), cv2.FONT_HERSHEY_PLAIN, 1, (200,200,200), 1)
                    cv2.putText(display, f"{int(track.velocity)}", (1060, y), cv2.FONT_HERSHEY_PLAIN, 1, (200,200,200), 1)
                    cv2.putText(display, threat, (1140, y), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
                    idx += 1
                
                if idx % 5 == 0: 
                    self.log_file.write(f"{curr_time:.2f},{tid},{int(azi_deg)},{int(rng_px)},{int(track.velocity)},{threat}\n")

            display[self.height-244:self.height-4, self.width-324:self.width-4] = (255, 255, 255)
            display[self.height-242:self.height-2, self.width-322:self.width-2] = small_frame
            
            cv2.imshow("ADX-C2 STATION", display)
            if cv2.waitKey(20) == 27: break
            
        self.cap.release()
        self.log_file.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    R = RadarSystem()
    R.run()
