import simpleGit from 'simple-git';
import moment from 'moment';

const git = simpleGit();

const autoCommitPush = async () => {
  try {
    // Stage all changes
    await git.add('./*');

    // Check if there are any changes to commit
    const status = await git.status();
    if (status.staged.length === 0) {
      console.log('✅ Nothing to commit. Working tree clean.');
      return;
    }

    // Create commit message with timestamp
    const timestamp = moment().format('YYYY-MM-DD HH:mm:ss');
    await git.commit(`Auto-commit on ${timestamp}`);
    console.log(`✅ Committed changes at ${timestamp}`);

    // Push to current branch (usually 'main' or 'master')
    await git.push();
    console.log('🚀 Changes pushed to GitHub!');
  } catch (err) {
    console.error('❌ Git operation failed:', err);
  }
};

autoCommitPush();
