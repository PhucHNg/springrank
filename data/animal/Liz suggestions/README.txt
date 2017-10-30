Description of Dataset:

The compressed .zip file contains 172 data matrices, which were manually entered from data tables and figures from 111 unique publications. The citation information for all publications can be found in the metadata.csv file.

Notes:

- We imposed several criteria for data selection. As a result, not all available data matrices from a given publication was included in the dataset. For example, we excluded data with fewer than 6 individuals and studies in which fewer than 2 interactions per dyad was observed on average. In cases where (1) observed groups shared common members, (2) the same group was observed multiple times, or (3) different behaviors were recorded for the same group, we chose one data matrix (usually data presented first in the publication). 

- Data were presented in different formats across publications. In all cases, we have formatted the data into a N x N matrix in which the cell values represent the number of times the row individual ‘beats’ the column individual. Thus, in cases where the data represents direct aggression, the row individual displayed aggression towards the column individuals. However, where the data represents submissive displays, the row individual received the display from the column individual. 

Some notable special cases:

- Nakano (1995) observed interactions in groups that contained members of two different species of salmonids. We restricted our dataset to individuals of one species 

- Monnin & Peeters (1991) presented their data broken down by specific aggressive behaviors. We summed up all of the aggressive acts in the data matrix.

