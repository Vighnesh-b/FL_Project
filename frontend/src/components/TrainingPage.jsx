import React, { useState, useEffect } from 'react';
import { Play, RefreshCw, Activity, Users, Layers, TrendingUp, Server, AlertCircle, CheckCircle } from 'lucide-react';

const TRAINING_API = 'http://localhost:5001/api/training';

export default function TrainingPage() {
  const [status, setStatus] = useState(null);
  const [logs, setLogs] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [aggregations, setAggregations] = useState([]);
  const [isStarting, setIsStarting] = useState(false);

  useEffect(() => {
    fetchStatus();
    const statusInterval = setInterval(fetchStatus, 2000);
    const logsInterval = setInterval(fetchLogs, 3000);
    const metricsInterval = setInterval(fetchMetrics, 5000);
    
    return () => {
      clearInterval(statusInterval);
      clearInterval(logsInterval);
      clearInterval(metricsInterval);
    };
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await fetch(`${TRAINING_API}/status`);
      const data = await response.json();
      setStatus(data);
    } catch (err) {
      console.error('Failed to fetch status:', err);
    }
  };

  const fetchLogs = async () => {
    try {
      const response = await fetch(`${TRAINING_API}/logs?limit=100`);
      const data = await response.json();
      setLogs(data.logs);
    } catch (err) {
      console.error('Failed to fetch logs:', err);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${TRAINING_API}/metrics`);
      const data = await response.json();
      setMetrics(data.metrics_by_client);
      
      const aggResponse = await fetch(`${TRAINING_API}/aggregations`);
      const aggData = await aggResponse.json();
      setAggregations(aggData.aggregations);
    } catch (err) {
      console.error('Failed to fetch metrics:', err);
    }
  };
  const stopTraining = async () => {
    if (!window.confirm("Stop training immediately?")) return;
  
    try {
      await fetch(`${TRAINING_API}/stop`, { method: "POST" });
      setStatus({ ...status, is_training: false });
    } catch (err) {
      alert("Failed to stop training.");
    }
  };
  const startTraining = async () => {
    setIsStarting(true);
    try {
      const response = await fetch(`${TRAINING_API}/start`, { method: 'POST' });
      const data = await response.json();
      if (data.success) {
        setTimeout(fetchStatus, 1000);
        setTimeout(fetchLogs, 1000);
      } else {
        alert(data.error);
      }
    } catch (err) {
      alert('Failed to start training. Make sure training_api.py is running on port 5001!');
    } finally {
      setIsStarting(false);
    }
  };

    const resetTraining = async () => {
    if (!window.confirm('Reset training state and clear logs?')) return;
    try {
      await fetch(`${TRAINING_API}/reset`, { method: 'POST' });
      setLogs([]);
      setMetrics({});
      setAggregations([]);
      fetchStatus();
    } catch (err) {
      alert('Failed to reset');
    }
  };

  const getLogIcon = (type) => {
    switch (type) {
      case 'error': return <AlertCircle className="w-4 h-4 text-red-600" />;
      case 'aggregation': return <Server className="w-4 h-4 text-purple-600" />;
      case 'train': return <Activity className="w-4 h-4 text-blue-600" />;
      default: return <CheckCircle className="w-4 h-4 text-green-600" />;
    }
  };

  const getLogColor = (type) => {
    switch (type) {
      case 'error': return 'bg-red-50 border-red-200';
      case 'aggregation': return 'bg-purple-50 border-purple-200';
      case 'train': return 'bg-blue-50 border-blue-200';
      default: return 'bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-8 shadow-lg">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center gap-3 mb-2">
            <Users className="w-10 h-10" />
            <h1 className="text-4xl font-bold">Federated Learning Training</h1>
          </div>
          <p className="text-indigo-100 text-lg">
            Automated Sequential Multi-Client Training with Real-Time Monitoring
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Control Panel */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold text-gray-800">Training Control Panel</h2>
            <div className="flex gap-3">
              <button
                onClick={resetTraining}
                disabled={status?.is_training}
                className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
              >
                <RefreshCw className="w-4 h-4" />
                Reset
              </button>
              <button
                onClick={stopTraining}
                disabled={!status?.is_training}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-md hover:shadow-lg"
              >
              <AlertCircle className="w-4 h-4" />
                Stop
              </button>
              <button
                onClick={startTraining}
                disabled={status?.is_training || isStarting}
                className="px-6 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg hover:from-indigo-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg flex items-center gap-2"
              >
                {isStarting ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    Starting...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Start Training
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Status Cards */}
          {status && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 p-4 rounded-lg border border-indigo-200">
                  <div className="flex items-center gap-2 mb-1">
                    <Layers className="w-4 h-4 text-indigo-600" />
                    <span className="text-sm font-medium text-gray-700">Round</span>
                  </div>
                  <p className="text-2xl font-bold text-indigo-600">
                    {status.current_round} / {status.total_rounds}
                  </p>
                </div>

                <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg border border-purple-200">
                  <div className="flex items-center gap-2 mb-1">
                    <Users className="w-4 h-4 text-purple-600" />
                    <span className="text-sm font-medium text-gray-700">Client</span>
                  </div>
                  <p className="text-2xl font-bold text-purple-600">
                    {status.current_client} / {status.total_clients}
                  </p>
                </div>

                <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg border border-blue-200">
                  <div className="flex items-center gap-2 mb-1">
                    <Activity className="w-4 h-4 text-blue-600" />
                    <span className="text-sm font-medium text-gray-700">Epoch</span>
                  </div>
                  <p className="text-2xl font-bold text-blue-600">
                    {status.current_epoch}
                  </p>
                </div>

                <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg border border-green-200">
                  <div className="flex items-center gap-2 mb-1">
                    <TrendingUp className="w-4 h-4 text-green-600" />
                    <span className="text-sm font-medium text-gray-700">Progress</span>
                  </div>
                  <p className="text-2xl font-bold text-green-600">
                    {status.progress_percentage}%
                  </p>
                </div>
              </div>

              {/* Progress Bar */}
              {status.is_training && (
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-gradient-to-r from-indigo-600 to-purple-600 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${status.progress_percentage}%` }}
                  ></div>
                </div>
              )}
            </>
          )}

          {/* Error Display */}
          {status?.error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <p className="text-red-800 text-sm">{status.error}</p>
            </div>
          )}
        </div>

        {/* Logs and Metrics Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Real-Time Logs */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Training Logs
            </h2>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {logs.length === 0 ? (
                <p className="text-gray-500 text-sm text-center py-8">
                  No logs yet. Start training to see progress.
                </p>
              ) : (
                logs.slice().reverse().map((log, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-lg border ${getLogColor(log.type)} text-sm`}
                  >
                    <div className="flex items-start gap-2">
                      {getLogIcon(log.type)}
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-medium text-gray-800">{log.message}</span>
                          <span className="text-xs text-gray-500">{log.timestamp}</span>
                        </div>
                        {log.metrics && (
                          <div className="flex gap-4 text-xs text-gray-600 mt-1">
                            <span>Loss: {log.metrics.train_loss?.toFixed(4)}</span>
                            <span>Val Loss: {log.metrics.val_loss?.toFixed(4)}</span>
                            <span className="font-semibold text-blue-600">
                              Dice: {log.metrics.val_dice_percentage}%
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Client Metrics */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Client Metrics
            </h2>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {Object.keys(metrics).length === 0 ? (
                <p className="text-gray-500 text-sm text-center py-8">
                  No metrics yet. Metrics will appear as training progresses.
                </p>
              ) : (
                Object.entries(metrics).map(([clientId, clientMetrics]) => (
                  <div key={clientId} className="border border-gray-200 rounded-lg p-3">
                    <h3 className="font-semibold text-gray-800 mb-2">Client {clientId}</h3>
                    <div className="space-y-1">
                      {clientMetrics.slice(-3).map((metric, idx) => (
                        <div key={idx} className="text-xs text-gray-600 flex justify-between">
                          <span>R{metric.round} E{metric.epoch}</span>
                          <span>Loss: {metric.train_loss?.toFixed(4)}</span>
                          <span className="font-semibold text-blue-600">
                            Dice: {metric.val_dice_percentage}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Aggregation History */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
            <Server className="w-5 h-5" />
            Aggregation History
          </h2>
          <div className="overflow-x-auto">
            {aggregations.length === 0 ? (
              <p className="text-gray-500 text-sm text-center py-8">
                No aggregations yet. Aggregations will appear after each round completes.
              </p>
            ) : (
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-2 px-4 font-semibold text-gray-700">Round</th>
                    <th className="text-left py-2 px-4 font-semibold text-gray-700">Timestamp</th>
                    <th className="text-left py-2 px-4 font-semibold text-gray-700">Clients</th>
                    <th className="text-left py-2 px-4 font-semibold text-gray-700">Model Path</th>
                  </tr>
                </thead>
                <tbody>
                  {aggregations.map((agg, idx) => (
                    <tr key={idx} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-2 px-4">
                        <span className="inline-block bg-purple-100 text-purple-700 px-2 py-1 rounded text-sm font-medium">
                          Round {agg.round}
                        </span>
                      </td>
                      <td className="py-2 px-4 text-sm text-gray-600">{agg.timestamp}</td>
                      <td className="py-2 px-4 text-sm text-gray-600">{agg.num_clients}</td>
                      <td className="py-2 px-4 text-xs text-gray-500 font-mono">{agg.model_path}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}