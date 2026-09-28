const superagent = require('superagent');
const csv = require('csv-parser');
const fs = require('fs');
const cliProgress = require('cli-progress');

interface DatasetRecord {
    pic_url: string;
    cate_id: string;
    item_id: string;
}

// Define parallel downloaders
let currParallelDownloaders = 0;

// Define max parallel downloaders
const maxParallelDownloaders = 10000;

// Define progress bar
const progressBar = new cliProgress.SingleBar({
  format: 'Downloading [{bar}] {percentage}% | ETA: {eta_formatted} | Elapsed: {duration_formatted} | {value}/{total}',
  formatTime: (t: number) => formatDuration(t)
}, cliProgress.Presets.legacy);

// Read dataset.csv
const results: DatasetRecord[] = [];
fs.createReadStream('./files/items_url.csv')
  .pipe(csv())
  .on('data', (data: any) => results.push(data))
  .on('end', () => processDataset(results));


// Process dataset
async function processDataset(dataset: DatasetRecord[]) {

  // Get dataset length
  console.log("Total dataset records: ", dataset.length);
  progressBar.start(dataset.length, 0);

  // Create path for images
  if (!fs.existsSync('./img')) fs.mkdirSync('./img');

  // Async download of images in the dataset
  const download = async (url: string, path: string) => {

    // Check if image exists, if not download it
    if (!fs.existsSync(path)) {

      // Download image
      let response: any;
      while (true) {
        try {
          // response = await superagent.get(url);
          response = await superagent.get(url).buffer(true).parse(superagent.parse.image);
          if (response && response.status === 200) break;
          else await sleep(1000);
        } catch (e) {
          await sleep(1000);
        }
      }

      // Save buffer to file
      fs.appendFileSync(path, response.body);

      // Decrement parallel downloaders
      currParallelDownloaders--;
      progressBar.increment();

    } else {
      currParallelDownloaders--;
      progressBar.increment();
    }

  };

  // Download images in the dataset
  for (let i = 0; i < dataset.length; i++) {

    // Increment parallel downloaders
    currParallelDownloaders++;

    // Wait some seconds before starting a new batch
    while (currParallelDownloaders > maxParallelDownloaders) {
      await sleep(1000);
    }

    // Define path and url
    const record = dataset[i];
    const url = record.pic_url;
    const path = `./img/${record.item_id}.png`;


    // Download image
    download(url, path);

  }

  // Stop progress bar
  progressBar.stop();

}

// Define a sleep function
function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Define a function to format time
function formatDuration(t: number) {

  // Show days, hours, minutes and seconds based on time
  const days = Math.floor(t / 86400);
  const hours = Math.floor((t % 86400) / 3600);
  const minutes = Math.floor(((t % 86400) % 3600) / 60);
  const seconds = Math.floor(((t % 86400) % 3600) % 60);

  // Return formatted time
  return `${days}d ${hours}h ${minutes}m ${seconds}s`;

}