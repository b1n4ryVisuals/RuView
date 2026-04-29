"""
TouchDesigner OSC Bridge for WiFi-DensePose
============================================
Connects to the sensing server WebSocket and REST API, then forwards
real-time data to TouchDesigner via OSC UDP.

Install dependencies:
    pip install websockets python-osc aiohttp

Usage:
    python td_osc_bridge.py
    python td_osc_bridge.py --td-ip 192.168.1.50 --td-port 9000
    python td_osc_bridge.py --server http://192.168.1.10:3000
    python td_osc_bridge.py --room-width 6.0 --room-depth 4.0

Room coordinate system (--room-width W, --room-depth D):
    X axis: -W/2 (left wall) to +W/2 (right wall), 0 = room centre
    Y axis:  0 (near wall) to +D (far wall)
    Z axis:  0 to ~2.0 (estimated height in metres, 0 if not available)

    Default room is 5.0 m × 5.0 m.  Set these to your actual room size for
    accurate metre values in TD.  The sensing server generates poses on a
    640×480 pixel canvas; this bridge maps that to room metres automatically.

OSC addresses sent to TouchDesigner:
    /ruview/persons/count           int     total persons detected
    /ruview/presence                int     1 = someone present, 0 = empty
    /ruview/person/{id}/x           float   torso X position (metres, 0 = room centre)
    /ruview/person/{id}/y           float   torso Y position (metres, 0 = near wall)
    /ruview/person/{id}/z           float   torso Z / height estimate (metres)
    /ruview/person/{id}/confidence  float   detection confidence 0.0-1.0
    /ruview/person/{id}/activity    str     "still" | "moving"
    /ruview/vitals/breathing_bpm    float   breaths per minute (0 if no data)
    /ruview/vitals/heart_bpm        float   beats per minute (0 if no data)
    /ruview/vitals/fall_detected    int     1 if fall detected
    /ruview/node/{id}/rssi          float   RSSI dBm for each ESP32 node
    /ruview/node/{id}/status        str     "active" | "inactive"
    /ruview/signal/motion_power     float   motion band energy
    /ruview/signal/breathing_power  float   breathing band energy
"""

import argparse
import asyncio
import json
import logging

import aiohttp
import websockets
from pythonosc.udp_client import SimpleUDPClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("td_osc_bridge")

# Keypoint indices for torso centre (average of shoulders + hips)
_TORSO_KP_INDICES = {
    "left_shoulder": 5, "right_shoulder": 6,
    "left_hip": 11,     "right_hip": 12,
}

# Canvas dimensions used by the sensing server's signal_derived pose generator.
# Keypoint x ∈ [0, CANVAS_W], y ∈ [0, CANVAS_H].
_CANVAS_W = 640.0
_CANVAS_H = 480.0


def _torso_centre(keypoints: list) -> tuple[float, float, float]:
    """Return (pixel_x, pixel_y, z) torso centre from 17-keypoint list.

    Uses shoulders + hips (indices 5, 6, 11, 12); falls back to (0,0,0) if
    none have confidence > 0.1.  Pixel coords are normalised to room metres
    by the caller via _px_to_room().
    """
    xs, ys, zs = [], [], []
    for _, idx in _TORSO_KP_INDICES.items():
        if idx < len(keypoints):
            kp = keypoints[idx]
            conf = kp.get("confidence", 0)
            if conf >= 0.1:
                xs.append(kp.get("x", 0.0))
                ys.append(kp.get("y", 0.0))
                zs.append(kp.get("z", 0.0))
    if not xs:
        return 0.0, 0.0, 0.0
    return sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)


def _px_to_room(px: float, py: float, room_width: float, room_depth: float) -> tuple[float, float]:
    """Map pixel canvas coords to room metres.

    pixel x [0, 640] → room x [-room_width/2, +room_width/2]  (0 = centre)
    pixel y [0, 480] → room y [0, room_depth]                  (0 = near wall)
    """
    room_x = (px / _CANVAS_W - 0.5) * room_width
    room_y = (py / _CANVAS_H) * room_depth
    return room_x, room_y


class TDOscBridge:
    def __init__(self, ws_url: str, http_url: str, td_ip: str, td_port: int,
                 vitals_interval: float = 1.0,
                 room_width: float = 5.0, room_depth: float = 5.0,
                 max_persons: int = 3):
        self.ws_url = ws_url
        self.http_url = http_url
        self.td_ip = td_ip
        self.td_port = td_port
        self.vitals_interval = vitals_interval
        self.room_width = room_width
        self.room_depth = room_depth
        self.max_persons = max_persons
        self._osc = SimpleUDPClient(td_ip, td_port)
        self._last_vitals: dict = {}
        self._running = False
        # Stable slot mapping: server ID → fixed slot (1..max_persons)
        self._id_to_slot: dict[int, int] = {}
        self._active_slots: set[int] = set()

    def _send(self, address: str, *values):
        try:
            self._osc.send_message(address, list(values) if len(values) > 1 else values[0])
        except Exception as e:
            log.warning("OSC send failed %s: %s", address, e)

    # ------------------------------------------------------------------
    # WebSocket handler — pose + presence + signal features
    # ------------------------------------------------------------------

    async def _ws_loop(self):
        while self._running:
            try:
                log.info("Connecting to sensing WebSocket: %s", self.ws_url)
                async with websockets.connect(self.ws_url, ping_interval=20) as ws:
                    log.info("WebSocket connected — streaming to TD at %s:%d",
                             self.td_ip, self.td_port)
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if msg.get("type") == "pose_data":
                            self._handle_pose(msg)
            except Exception as e:
                if self._running:
                    log.warning("WebSocket error: %s — reconnecting in 3s", e)
                    await asyncio.sleep(3)

    def _handle_pose(self, msg: dict):
        payload = msg.get("payload", {})
        pose = payload.get("pose", {})
        persons = pose.get("persons", [])
        meta = payload.get("metadata", {})

        # Person count + presence
        count = len(persons)
        self._send("/ruview/persons/count", count)
        self._send("/ruview/presence", 1 if count > 0 else 0)

        # Assign stable slots to incoming server IDs so TD channels don't multiply.
        # Slots are 1..max_persons and reused when persons drop out.
        activity = payload.get("activity", "still")
        capped = persons[: self.max_persons]
        current_slots: set[int] = set()

        for person in capped:
            sid = person.get("id", 1)
            # Assign a slot if this server ID is new
            if sid not in self._id_to_slot:
                used = set(self._id_to_slot.values())
                free = next((s for s in range(1, self.max_persons + 1) if s not in used), None)
                if free is None:
                    continue  # all slots full — skip extra detections
                self._id_to_slot[sid] = free
            slot = self._id_to_slot[sid]
            current_slots.add(slot)

            kps = person.get("keypoints", [])
            if kps:
                px, py, kp_z = _torso_centre(kps)
                if px == 0.0 and py == 0.0:
                    bbox = person.get("bbox", {})
                    px = bbox.get("x", 320.0) + bbox.get("width", 0.0) / 2.0
                    py = bbox.get("y", 240.0) + bbox.get("height", 0.0) / 2.0
                    kp_z = 0.0
            else:
                bbox = person.get("bbox", {})
                px = bbox.get("x", 320.0) + bbox.get("width", 0.0) / 2.0
                py = bbox.get("y", 240.0) + bbox.get("height", 0.0) / 2.0
                kp_z = 0.0

            room_x, room_y = _px_to_room(px, py, self.room_width, self.room_depth)
            conf = person.get("confidence", 0.0)
            self._send(f"/ruview/person/{slot}/x", float(room_x))
            self._send(f"/ruview/person/{slot}/y", float(room_y))
            self._send(f"/ruview/person/{slot}/z", float(kp_z))
            self._send(f"/ruview/person/{slot}/confidence", float(conf))
            self._send(f"/ruview/person/{slot}/activity", str(activity))

        # Clear slots whose persons are no longer in the frame
        dropped = self._active_slots - current_slots
        for slot in dropped:
            self._send(f"/ruview/person/{slot}/x", 0.0)
            self._send(f"/ruview/person/{slot}/y", 0.0)
            self._send(f"/ruview/person/{slot}/z", 0.0)
            self._send(f"/ruview/person/{slot}/confidence", 0.0)
            self._send(f"/ruview/person/{slot}/activity", "gone")
        # Release server ID → slot mappings for dropped persons
        dropped_sids = [sid for sid, s in self._id_to_slot.items() if s in dropped]
        for sid in dropped_sids:
            del self._id_to_slot[sid]
        self._active_slots = current_slots

        # Signal features
        self._send("/ruview/signal/motion_power",
                   float(meta.get("motion_band_power", 0.0)))
        self._send("/ruview/signal/breathing_power",
                   float(meta.get("breathing_band_power", 0.0)))

        # Forward cached vitals every pose frame so TD values stay current
        if self._last_vitals:
            self._send_vitals(self._last_vitals)

    # ------------------------------------------------------------------
    # REST polling — vitals + node health
    # ------------------------------------------------------------------

    async def _poll_loop(self):
        async with aiohttp.ClientSession() as session:
            while self._running:
                await asyncio.gather(
                    self._poll_vitals(session),
                    self._poll_nodes(session),
                    return_exceptions=True,
                )
                await asyncio.sleep(self.vitals_interval)

    async def _poll_vitals(self, session: aiohttp.ClientSession):
        try:
            async with session.get(f"{self.http_url}/api/v1/edge-vitals",
                                   timeout=aiohttp.ClientTimeout(total=2)) as r:
                if r.status == 200:
                    data = await r.json()
                    v = data.get("edge_vitals")
                    if v:
                        self._last_vitals = v
                        self._send_vitals(v)
        except Exception:
            pass

    def _send_vitals(self, v: dict):
        self._send("/ruview/vitals/breathing_bpm",
                   float(v.get("breathing_bpm", 0.0)))
        self._send("/ruview/vitals/heart_bpm",
                   float(v.get("heart_bpm", 0.0)))
        self._send("/ruview/vitals/fall_detected",
                   int(bool(v.get("fall_detected", False))))

    async def _poll_nodes(self, session: aiohttp.ClientSession):
        try:
            async with session.get(f"{self.http_url}/api/v1/nodes",
                                   timeout=aiohttp.ClientTimeout(total=2)) as r:
                if r.status == 200:
                    data = await r.json()
                    for node in data.get("nodes", []):
                        nid = node.get("node_id", 0)
                        self._send(f"/ruview/node/{nid}/rssi",
                                   float(node.get("rssi_dbm", 0.0)))
                        self._send(f"/ruview/node/{nid}/status",
                                   str(node.get("status", "unknown")))
        except Exception:
            pass

    # ------------------------------------------------------------------

    async def run(self):
        self._running = True
        log.info("TD OSC Bridge started — sending to %s:%d  room=%.1fm × %.1fm",
                 self.td_ip, self.td_port, self.room_width, self.room_depth)
        await asyncio.gather(self._ws_loop(), self._poll_loop())

    def stop(self):
        self._running = False


def main():
    parser = argparse.ArgumentParser(description="WiFi-DensePose → TouchDesigner OSC Bridge")
    parser.add_argument("--server", default="http://localhost:3000",
                        help="Sensing server base URL (default: http://localhost:3000)")
    parser.add_argument("--td-ip", default="127.0.0.1",
                        help="TouchDesigner host IP (default: 127.0.0.1)")
    parser.add_argument("--td-port", type=int, default=9000,
                        help="TouchDesigner OSC in port (default: 9000)")
    parser.add_argument("--vitals-interval", type=float, default=1.0,
                        help="Vitals + node poll interval in seconds (default: 1.0)")
    parser.add_argument("--room-width", type=float, default=5.0,
                        help="Room width in metres — maps to X axis (default: 5.0)")
    parser.add_argument("--room-depth", type=float, default=5.0,
                        help="Room depth in metres — maps to Y axis (default: 5.0)")
    parser.add_argument("--max-persons", type=int, default=3,
                        help="Max simultaneous persons to track (default: 3)")
    args = parser.parse_args()

    http_url = args.server.rstrip("/")
    # /api/v1/stream/pose is the endpoint that wraps sensing data as pose_data messages.
    # /ws/sensing on port 8765 sends raw sensing_update messages in a different format.
    ws_url = http_url.replace("http://", "ws://").replace("https://", "wss://") + "/api/v1/stream/pose"
    bridge = TDOscBridge(
        ws_url=ws_url,
        http_url=http_url,
        td_ip=args.td_ip,
        td_port=args.td_port,
        vitals_interval=args.vitals_interval,
        room_width=args.room_width,
        room_depth=args.room_depth,
        max_persons=args.max_persons,
    )
    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        bridge.stop()
        log.info("Bridge stopped.")


if __name__ == "__main__":
    main()
