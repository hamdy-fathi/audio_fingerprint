'use client';
import { useRef, useState, useEffect } from 'react';
import { Play, Pause } from 'lucide-react';
import { getFileUrl } from '@/lib/api';

export default function AudioPlayer({ fileId, onTimeUpdate, onPlay, onPause }) {
  const audioRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (audioRef.current && fileId) {
      audioRef.current.src = getFileUrl(fileId);
      audioRef.current.load();
      setPlaying(false);
      setCurrent(0);
    }
  }, [fileId]);

  const toggle = () => {
    if (!audioRef.current) return;
    if (playing) { audioRef.current.pause(); onPause?.(); }
    else { audioRef.current.play(); onPlay?.(); }
    setPlaying(!playing);
  };

  const seek = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    if (audioRef.current) {
      audioRef.current.currentTime = pct * duration;
    }
  };

  const fmt = (s) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  return (
    <div className="audio-player">
      <audio
        ref={audioRef}
        onTimeUpdate={() => {
          const t = audioRef.current?.currentTime || 0;
          setCurrent(t);
          onTimeUpdate?.(t);
        }}
        onLoadedMetadata={() => setDuration(audioRef.current?.duration || 0)}
        onEnded={() => { setPlaying(false); onPause?.(); }}
      />
      <button className="play-btn" onClick={toggle} disabled={!fileId} id="audio-play-btn">
        {playing ? <Pause size={18} /> : <Play size={18} />}
      </button>
      <div className="player-progress" onClick={seek}>
        <div className="player-progress-fill" style={{ width: duration ? `${(currentTime / duration) * 100}%` : '0%' }} />
      </div>
      <span className="player-time">{fmt(currentTime)} / {fmt(duration)}</span>
    </div>
  );
}
