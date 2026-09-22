/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        cyber: {
          blue: '#00f0ff',
          purple: '#9d4edd',
          red: '#ff0055',
          green: '#00ff88',
        }
      }
    },
  },
  plugins: [],
}
