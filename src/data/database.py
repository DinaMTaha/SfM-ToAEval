import sqlite3
import compress_pickle as pickle


class Database:
    def __init__(self, database_path):
        self.database_path = database_path
        self.con = sqlite3.connect(database_path)
        self.cur = self.con.cursor()
        self.cur.executescript('''
        CREATE TABLE IF NOT EXISTS sequence (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE);
        CREATE TABLE IF NOT EXISTS feature_type (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE);
        CREATE TABLE IF NOT EXISTS image (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            sequence INTEGER NOT NULL,
            FOREIGN KEY(sequence) REFERENCES sequence(id) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS image_feature (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            image INTEGER NOT NULL,
            detector_type INTEGER NOT NULL,
            feature_type INTEGER NOT NULL,
            count INTEGER,
            size REAL,
            time REAL,
            data BLOB,
            FOREIGN KEY(image) REFERENCES image(id) ON DELETE CASCADE,
            FOREIGN KEY(detector_type) REFERENCES feature_type(id) ON DELETE CASCADE,
            FOREIGN KEY(feature_type) REFERENCES feature_type(id) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS feature_match (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            feature1 INTEGER NOT NULL,
            feature2 INTEGER NOT NULL,
            count INTEGER,
            time REAL,
            data BLOB,
            FOREIGN KEY(feature1) REFERENCES image_feature(id) ON DELETE CASCADE,
            FOREIGN KEY(feature2) REFERENCES image_feature(id) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            sequence INTEGER NOT NULL,
            detector_type INTEGER NOT NULL,
            feature_type INTEGER NOT NULL,
            points INTEGER,
            observations INTEGER,
            error REAL,
            FOREIGN KEY(sequence) REFERENCES sequence(id) ON DELETE CASCADE,
            FOREIGN KEY(detector_type) REFERENCES feature_type(id) ON DELETE CASCADE,
            FOREIGN KEY(feature_type) REFERENCES feature_type(id) ON DELETE CASCADE);
        ''')

    def commit(self):
        if self.con:
            self.con.commit()

    def close(self):
        if self.con:
            self.con.commit()
            self.con.close()

    def to_blob(self, data):
        return sqlite3.Binary(pickle.dumps(data, compression='lzma'))

    def from_blob(self, blob):
        return pickle.loads(blob, compression='lzma')

    def getSequenceId(self, name):
        self.cur.execute('SELECT id from sequence WHERE name = ?', (name,))
        row = self.cur.fetchone()
        if row:
            return row[0]
        else:
            self.cur.execute('INSERT INTO sequence(name) VALUES (?)', (name,))
            return self.cur.lastrowid

    def getFeatureTypeId(self, name):
        self.cur.execute('SELECT id from feature_type WHERE name = ?', (name,))
        row = self.cur.fetchone()
        if row:
            return row[0]
        else:
            self.cur.execute(
                'INSERT INTO feature_type(name) VALUES (?)', (name,))
            return self.cur.lastrowid

    def getImageId(self, name, sequence):
        self.cur.execute(
            'SELECT id from image WHERE name = ? AND sequence = ?', (name, sequence))
        row = self.cur.fetchone()
        if row:
            return row[0]
        else:
            self.cur.execute(
                'INSERT INTO image(name, sequence) VALUES (?, ?)', (name, sequence))
            return self.cur.lastrowid

    def getFeature(self, image, detector_type, feature_type):
        self.cur.execute('SELECT id, data from image_feature WHERE image = ? AND detector_type = ? AND feature_type = ?',
                         (image, detector_type, feature_type))
        row = self.cur.fetchone()
        if row:
            return row[0], self.from_blob(row[1])
        else:
            return None

    def putFeature(self, image, detector_type, feature_type, count, size, time, data):
        self.cur.execute('INSERT INTO image_feature(image, detector_type, feature_type, count, size, time, data) VALUES (?, ?, ?, ?, ?, ?, ?)',
                         (image, detector_type, feature_type, count, size, time, self.to_blob(data)))
        return self.cur.lastrowid

    def getMatch(self, feature1, feature2):
        self.cur.execute(
            'SELECT id, data from feature_match WHERE feature1 = ? AND feature2 = ?', (feature1, feature2))
        row = self.cur.fetchone()
        if row:
            return row[0], self.from_blob(row[1])
        else:
            return None

    def putMatch(self, feature1, feature2, count, time, data):
        self.cur.execute('INSERT INTO feature_match(feature1, feature2, count, time, data) VALUES (?, ?, ?, ?, ?)',
                         (feature1, feature2, count, time, self.to_blob(data)))
        return self.cur.lastrowid
