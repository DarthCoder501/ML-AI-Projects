import React, { useState } from 'react';

const UserProfile = () => {
  const [experience, setExperience] = useState('');
  const [position, setPosition] = useState('');
  const [goals, setGoals] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    // Handle form submission
    console.log({ experience, position, goals });
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>
        Experience:
        <input
          type="text"
          value={experience}
          onChange={(e) => setExperience(e.target.value)}
        />
      </label>
      <label>
        Position:
        <input
          type="text"
          value={position}
          onChange={(e) => setPosition(e.target.value)}
        />
      </label>
      <label>
        Goals:
        <textarea
          value={goals}
          onChange={(e) => setGoals(e.target.value)}
        />
      </label>
      <button type="submit">Save Profile</button>
    </form>
  );
};

export default UserProfile;