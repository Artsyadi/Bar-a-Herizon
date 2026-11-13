"""atd-cleaner: clean automatic tracking data (players + ball) for any match, by code only.

Follows the team's merged Rulebook (sections 1.7, 4.1, 4.2, 4.10, C6.1-C6.3, 3.5):
  * missing observations mean neither an exit nor a turnover - nothing here decides D, H or A
  * interpolate only short gaps with credible endpoints, compatible motion and no sign of an
    intervening interaction or discontinuity
  * keep observed, estimated and unknown values separate
  * duplicates and identity switches are data-quality problems, flagged and reported
  * missing speed is never replaced with zero

Every threshold lives in config/cleaner_thresholds.json and must be frozen on external matches.
"""

__version__ = "0.2.1"
