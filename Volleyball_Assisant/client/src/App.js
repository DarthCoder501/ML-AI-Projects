import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import Home from './pages/Home';
import UserProfile from './pages/UserProfile';
import TrainingPlan from './pages/TrainingPlan';
import ProgressTracking from './pages/ProgressTracking';

function App() {
  return (
    <Router>
      <Switch>
        <Route path="/" exact component={Home} />
        <Route path="/user-profile" component={UserProfile} />
        <Route path="/training-plan" component={TrainingPlan} />
        <Route path="/progress-tracking" component={ProgressTracking} />
      </Switch>
    </Router>
  );
}

export default App;