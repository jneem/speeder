const MIN_PITCH: u32 = 65;
const MAX_PITCH: u32 = 400;
const DOWNSAMPLE_FREQ: u32 = 4000;

pub struct Stream {
    input: Vec<i16>,
    output: Vec<i16>,
    downsample: Vec<i16>,
    sample_rate: u32,
    // TODO: set this to sample_rate / DOWNSAMPLE_FREQ
    skip_factor: u32,
    speed: f32,
    prev_period: u32,
    prev_min_diff: u64,
    max_period: u32,
    input_to_copy: usize,
}

impl Stream {
    pub fn new(sample_rate: u32, speed: f32) -> Stream {
        Stream {
            input: Vec::new(),
            output: Vec::new(),
            downsample: Vec::new(), // TODO: can predict the size and pre-allocate
            sample_rate,
            skip_factor: sample_rate / DOWNSAMPLE_FREQ,
            speed,
            prev_period: 0,
            prev_min_diff: 0,
            max_period: sample_rate / MAX_PITCH,
            input_to_copy: 0,
        }
    }

    pub fn add_samples(&mut self, samples: &[i16]) {
        self.input.extend_from_slice(samples);
    }

    pub fn read_samples<'buf>(&mut self, buf: &'buf mut [i16]) -> &'buf mut [i16] {
        let count = usize::max(self.output.len(), buf.len());
        for (y, x) in buf.iter_mut().zip(self.output.drain(..count)) {
            *y = x;
        }
        &mut buf[..count]
    }

    // In order to do anything, we need a buffer of at least this size (twice the max period).
    fn samples_required(&self) -> usize {
        self.max_period as usize * 2
    }

    fn downsample(&mut self, offset: usize) {
        assert!(self.input[offset..].len() >= self.samples_required() as usize);

        self.downsample.clear();
        let count = self.samples_required() as u32 / self.skip_factor;
        for chunk in self.input[offset..]
            .chunks_exact(self.skip_factor as usize)
            .take(count as usize)
        {
            let mean: i32 = chunk.iter().map(|&x| x as i32).sum::<i32>() / self.skip_factor as i32;
            self.downsample.push(mean as i16);
        }
    }

    fn min_period(&self) -> u32 {
        self.sample_rate as u32 / MAX_PITCH
    }

    fn max_period(&self) -> u32 {
        self.sample_rate as u32 / MIN_PITCH
    }

    fn detect_pitch(&mut self, offset: usize) -> u32 {
        // First, downsample and estimate the pitch of the downsampled signal.
        self.downsample(offset);
        let ds_result = pitch_period(
            &self.downsample[..],
            self.min_period() / self.skip_factor,
            self.max_period() / self.skip_factor,
        );

        // Now search for the pitch in the original signal, but using the downsampled pitch
        // to narrow down the search.
        let ds_period = ds_result.period * self.skip_factor;
        let min_period = self.min_period().max(ds_period - 4 * self.skip_factor);
        let max_period = self.max_period().min(ds_period + 4 * self.skip_factor);

        let result = pitch_period(&self.input[offset..], min_period, max_period);
        let ret = self.consider_prev_period(&result);

        self.prev_min_diff = result.min_diff;
        self.prev_period = result.period;
        ret
    }

    fn consider_prev_period(&self, res: &Period) -> u32 {
        if res.min_diff == 0 || self.prev_period == 0 {
            res.period
        } else if res.max_diff > 3 * res.min_diff {
            res.period
        } else if 2 * res.min_diff <= 3 * self.prev_min_diff {
            res.period
        } else {
            self.prev_period
        }
    }

    fn skip_pitch_period(&mut self, input_offset: usize, period: u32) -> usize {
        let new_samples = if self.speed >= 2.0 {
            (period as f32 / (self.speed - 1.0)) as usize
        } else {
            self.input_to_copy = (period as f32 * (2.0 - self.speed) / (self.speed - 1.0)) as usize;
            period as usize
        };

        let input = &self.input[input_offset..];
        overlap_add_append(
            &mut self.output,
            &input[..new_samples],
            &input[period as usize..(period as usize + new_samples)],
        );
        new_samples
    }

    fn insert_pitch_period(&mut self, input_offset: usize, period: u32) -> usize {
        let speed = self.speed;
        let new_samples = if speed < 0.5 {
            (period as f32 * speed / (1.0 - speed)) as usize
        } else {
            self.input_to_copy = (period as f32 * (2.0 * speed - 1.0) / (1.0 - speed)) as usize;
            period as usize
        };

        let input = &self.input[input_offset..];
        let period = period as usize;
        self.output.extend_from_slice(&input[..period]);
        overlap_add_append(
            &mut self.output,
            &input[period..(period + new_samples)],
            &input[..new_samples],
        );
        new_samples
    }

    pub fn process(&mut self) {
        let mut input_offset = 0;
        while input_offset + self.samples_required() < self.input.len() {
            if self.input_to_copy > 0 {
                // FIXME: we don't actually need `samples_required` more samples here.
                self.output.extend_from_slice(
                    &self.input[input_offset..(input_offset + self.input_to_copy as usize)],
                );
                input_offset += self.input_to_copy;
                self.input_to_copy = 0;
            } else {
                let period = self.detect_pitch(input_offset);
                if self.speed > 1.0 {
                    let consumed = self.skip_pitch_period(input_offset, period);
                    input_offset += period as usize + consumed;
                } else {
                    let consumed = self.insert_pitch_period(input_offset, period);
                    input_offset += consumed;
                }
            }
        }
    }

    pub fn output(&self) -> &[i16] {
        &self.output[..]
    }
}

struct Period {
    period: u32,
    max_diff: u64,
    min_diff: u64,
}

fn pitch_period(samples: &[i16], min_period: u32, max_period: u32) -> Period {
    let mut best_period = 0;
    let mut worst_period = 1;
    let mut best_diff = 0;
    let mut worst_diff = 0;
    let min_period = min_period as usize;
    let max_period = max_period as usize;

    for period in min_period..=max_period {
        let diff: usize = samples[..period]
            .iter()
            .zip(&samples[period..2 * period])
            .map(|(x, y)| (x - y).abs() as usize)
            .sum();

        if best_period == 0 || diff * best_period < best_diff * period {
            best_period = period;
            best_diff = diff;
        }
        if diff * worst_period > worst_diff * period {
            worst_diff = diff;
            worst_period = period;
        }
    }

    Period {
        period: best_period as u32,
        max_diff: (worst_diff / worst_period) as u64,
        min_diff: (best_diff / best_period) as u64,
    }
}

/// Zip 3 things.
pub fn zip3<I, J, K>(i: I, j: J, k: K) -> impl Iterator<Item = (I::Item, J::Item, K::Item)>
where
    I: IntoIterator,
    J: IntoIterator,
    K: IntoIterator,
{
    i.into_iter()
        .zip(j.into_iter().zip(k))
        .map(|(x, (y, z))| (x, y, z))
}

fn overlap_add_append(out: &mut Vec<i16>, down: &[i16], up: &[i16]) {
    assert_eq!(down.len(), up.len());
    let n = down.len();
    out.reserve(n);

    for (down, up, i) in zip3(down, up, 0..n) {
        let ratio = f32::sin(i as f32 * std::f32::consts::PI / (2.0 * n as f32));
        out.push((*down as f32 * (1.0 - ratio) + *up as f32 * ratio) as i16);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
